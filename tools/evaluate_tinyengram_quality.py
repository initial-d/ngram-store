"""CPU-only qualification of a local trained TinyEngram/Qwen memory branch.

Compare complete next-token distributions to memory-off, identity-free head
means and byte-preserving native lookup. This quality oracle retains source
memory and is explicitly unsuitable for deployment RSS/throughput comparisons.
No downloads, GPU use, training or compression selection on test text.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import torch
import transformers
from transformers import AutoTokenizer

from .engram_row_adapter import MappedMultiHeadEmbedding, file_sha256
from .lm_quality_metrics import memory_qualification, next_token_metrics, summarize_windows
from .tinyengram_checkpoint import TinyEngramModel, load_upstream, source_revision, UPSTREAM_SHA


def token_hash(ids):
    return hashlib.sha256(ids.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_windows(ids, length, calibration_windows, test_windows):
    if (ids.ndim != 2 or ids.shape[0] != 1 or length < 2
            or min(calibration_windows, test_windows) < 2
            or ids.shape[1] < length * (calibration_windows + test_windows)):
        raise ValueError("insufficient tokens or invalid nonoverlapping window protocol")
    return [("calibration" if i < calibration_windows else "test", i,
             ids[:, i * length:(i + 1) * length])
            for i in range(calibration_windows + test_windows)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "bundle", "corpus", "upstream-dir", "protocol", "report-out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--native-library", type=Path)
    args = p.parse_args()
    if args.report_out.exists():
        p.error("use a new report path; earlier evidence is immutable")
    protocol = json.loads(args.protocol.read_text())
    required = {"schema_version", "excluded_prefix_characters", "window_tokens", "calibration_windows",
                "test_windows", "threads", "minimum_ppl_effect", "corpus_sha256"}
    if not required <= set(protocol) or protocol["schema_version"] != 1:
        p.error("unsupported or incomplete frozen protocol")
    if protocol["excluded_prefix_characters"] < 16384 or protocol["threads"] < 1:
        p.error("exclude all text used by the reference training and use positive CPU threads")
    if not 0 < protocol["minimum_ppl_effect"] < 1:
        p.error("minimum PPL effect must be between zero and one")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "running", "command": shlex.join(sys.orig_argv),
              "code_sha": source_revision(), "upstream_sha": UPSTREAM_SHA,
              "protocol": protocol, "protocol_sha256": file_sha256(args.protocol),
              "source_file_hashes": {name: file_sha256(Path(__file__).parent / name) for name in
                                     ("evaluate_tinyengram_quality.py", "lm_quality_metrics.py",
                                      "tinyengram_checkpoint.py", "engram_row_adapter.py", "native_row_store.py")},
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "transformers": transformers.__version__, "os": platform.platform(),
                              "device": "cpu", "threads": protocol["threads"], "interop_threads": 1},
              "limitations": ["one 32-step checkpoint with 648 rows; no large-table capacity evidence",
                              "TinyShakespeare suffix is held out from branch training, not necessarily Qwen pretraining",
                              "inference ablations are not independently trained baseline models",
                              "fixed-window eager teacher forcing; no generation or deployment performance claim",
                              "paired-window bootstrap ignores longer-range text dependence and model-seed uncertainty",
                              "quality oracle retains original dense memory; wall time and RSS are not backend benchmarks"],
              "windows": {name: {split: [] for split in ("calibration", "test")}
                          for name in ("memory-off", "head-mean", "native-fp32")}}
    start = time.perf_counter()
    native = {}

    def save():
        temporary = args.report_out.with_suffix(".partial")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(args.report_out)

    try:
        torch.set_num_threads(protocol["threads"])
        torch.set_num_interop_threads(1)
        try:
            report["environment"]["cpu"] = subprocess.check_output(["lscpu"], text=True)
        except (OSError, subprocess.CalledProcessError):
            report["environment"]["cpu"] = platform.processor()
        if file_sha256(args.corpus) != protocol["corpus_sha256"]:
            raise ValueError("corpus differs from frozen protocol")
        report["checkpoint_files"] = {f.name: {"bytes": f.stat().st_size, "sha256": file_sha256(f)}
                                      for f in sorted(args.checkpoint.iterdir()) if f.is_file()}
        report["bundle_manifest_sha256"] = file_sha256(args.bundle / "bundle.json")
        engram, qwen = load_upstream(args.upstream_dir)
        values = json.loads((args.checkpoint / "engram_config.json").read_text())
        values["tokenizer_name_or_path"] = str(args.checkpoint.resolve())
        config = engram.EngramConfig(**values)
        step = json.loads((args.checkpoint / "trainer_state.json").read_text())["global_step"]
        engram.set_global_step(step)
        model, loading = qwen.EngramQwenForCausalLM.from_pretrained(
            args.checkpoint, engram_cfg=config, local_files_only=True,
            attn_implementation="eager", output_loading_info=True)
        if any(loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"incomplete checkpoint load: {loading}")
        wrapped = TinyEngramModel(model.requires_grad_(False)).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        text = args.corpus.read_text()[protocol["excluded_prefix_characters"]:]
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        windows = make_windows(ids, protocol["window_tokens"], protocol["calibration_windows"], protocol["test_windows"])
        report["split_token_hashes"] = {split: token_hash(torch.cat([x for s, _, x in windows if s == split], 1))
                                        for split in ("calibration", "test")}
        report["source_checkpoint_loading"] = loading
        manifest = json.loads((args.bundle / "bundle.json").read_text())
        original = {key: layer.multi_head_embedding for key, layer in model.model.engram_layers.items()}
        means = {}
        report["memory"] = {}
        for key, embedding in original.items():
            weight = embedding.embedding.weight.detach()
            if weight.dtype != torch.float32:
                raise ValueError("this qualification protocol requires original FP32 memory")
            rows = weight.clone()
            offsets = embedding.offsets.flatten().tolist() + [len(rows)]
            for begin, end in zip(offsets, offsets[1:]):
                rows[begin:end] = weight[begin:end].mean(0)
            means[key] = rows
            entry = manifest["layers"][key]
            native[key] = MappedMultiHeadEmbedding(args.bundle / entry["path"],
                          native_library=args.native_library, native_cache_bytes=1024**2)
            if native[key].table.rows(range(len(weight))) != weight.contiguous().view(torch.uint8).numpy().tobytes():
                raise AssertionError("native source memory bytes changed")
            if native[key].offsets.tolist() != embedding.offsets.tolist():
                raise AssertionError("native head offsets changed")
            report["memory"][key] = {"rows": len(weight), "dim": weight.shape[1], "raw_bytes": weight.numel()*4,
                                    "weight_sha256": token_hash(weight), "head_offsets": offsets[:-1],
                                    "native_files": {f.name: {"bytes": f.stat().st_size, "sha256": file_sha256(f)}
                                                     for f in (args.bundle / entry["path"]).iterdir() if f.is_file()}}
        with torch.inference_mode():
            for split, i, sample in windows:
                reference = wrapped(sample, use_cache=False).logits
                for name in report["windows"]:
                    backups = {}
                    try:
                        if name == "memory-off":
                            qwen.set_skip_engram(True)
                        elif name == "head-mean":
                            for key, embedding in original.items():
                                backups[key] = embedding.embedding.weight.clone()
                                embedding.embedding.weight.copy_(means[key])
                        else:
                            for key, layer in model.model.engram_layers.items():
                                layer.multi_head_embedding = native[key]
                        output = wrapped(sample, use_cache=False).logits
                        metrics = next_token_metrics(reference, output, sample)
                        if name == "native-fp32" and not metrics["logits_byte_exact"]:
                            raise AssertionError("native matched-schedule logits changed")
                        metrics["window_index"] = i
                        report["windows"][name][split].append(metrics)
                        del output
                    finally:
                        qwen.set_skip_engram(False)
                        for key, layer in model.model.engram_layers.items():
                            layer.multi_head_embedding = original[key]
                            if key in backups:
                                original[key].embedding.weight.copy_(backups[key])
                del reference
                report["elapsed_seconds"] = time.perf_counter() - start
                save()
                print(f"window {i+1}/{len(windows)} {split} elapsed={report['elapsed_seconds']:.1f}s", flush=True)
        report["summary"] = {split: {name: summarize_windows(values[split])
                                     for name, values in report["windows"].items()}
                             for split in ("calibration", "test")}
        report["memory_qualified_on_calibration"] = memory_qualification(
            report["summary"]["calibration"], minimum_ppl_effect=protocol["minimum_ppl_effect"])
        report["test_confirmation"] = memory_qualification(
            report["summary"]["test"], minimum_ppl_effect=protocol["minimum_ppl_effect"])
        report["status"] = "passed"
    except BaseException as error:
        report["status"], report["error"] = "failed", repr(error)
        raise
    finally:
        for embedding in native.values():
            embedding.close()
        report["elapsed_seconds"] = time.perf_counter() - start
        save()
        print("Report:", args.report_out.resolve(), flush=True)


if __name__ == "__main__":
    main()
