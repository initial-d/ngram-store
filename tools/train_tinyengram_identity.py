"""Actual-text TinyEngram branch training with frozen-prefix CPU/CUDA reuse.

Train ordinary next-token CE from a local pretrained Qwen. Freeze the protocol,
retain every final seed and both identity/branch ablations, and never choose a
checkpoint on test. Cached prefix arithmetic/gradients must match full Qwen.
The quality oracle retains source tables and is not a serving-memory benchmark.
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

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Config
import transformers

from .engram_row_adapter import MappedMultiHeadEmbedding, file_sha256
from .device_row_adapter import StagedMultiHeadEmbedding
from .frozen_tinyengram_prefix import FrozenTinyEngramPrefix, check_reuse
from .lm_quality_metrics import memory_qualification, next_token_metrics, summarize_windows
from .tinyengram_checkpoint import export_tinyengram_checkpoint, load_upstream, source_revision, UPSTREAM_SHA


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sample_windows(ids, length, count):
    available = ids.shape[1] // length
    if available < count or count < 1:
        raise ValueError("insufficient complete nonoverlapping training windows")
    selected = np.rint(np.linspace(0, available - 1, count)).astype(np.int64)
    if len(np.unique(selected)) != count:
        raise AssertionError("training sampling duplicated windows")
    return torch.stack([ids[0, i*length:(i+1)*length] for i in selected]), selected


def write_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate_seed(model, reuse, ids, prefixes, bundle, library):
    originals = {key: layer.multi_head_embedding for key, layer in model.model.engram_layers.items()}
    means, native = {}, {}
    result = {name: {s: [] for s in ids} for name in ("memory-off", "head-mean", "native-fp32")}
    manifest = json.loads((bundle / "bundle.json").read_text())
    try:
        for key, embedding in originals.items():
            weight = embedding.embedding.weight.detach()
            mean = weight.clone()
            offsets = embedding.offsets.tolist() + [len(weight)]
            for begin, end in zip(offsets, offsets[1:]):
                mean[begin:end] = weight[begin:end].mean(0)
            means[key] = mean
            path = bundle / manifest["layers"][key]["path"]
            if reuse.device.type == "cuda":
                native[key] = StagedMultiHeadEmbedding(path,device=reuse.device,native_library=library,
                              native_cache_bytes=1024**2,staging_rows=4096,staging_slots=2)
                gather = native[key].gather
            else:
                native[key] = MappedMultiHeadEmbedding(path,native_library=library,native_cache_bytes=1024**2)
                gather = native[key]
            if gather.table.rows(range(len(weight))) != weight.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes():
                raise AssertionError("export changed a source row")
            if not torch.equal(gather.offsets, embedding.offsets.cpu()):
                raise AssertionError("export changed original head offsets")
        with torch.no_grad():
            for split, windows in ids.items():
                for i, sample in enumerate(windows):
                    sample = sample.unsqueeze(0)
                    hidden = torch.from_numpy(np.array(prefixes[split][i:i+1], copy=True)).to(reuse.device)
                    reference = reuse.forward(sample, hidden)
                    # Validate a direct complete forward in each independently
                    # prepared split, in addition to the pretraining gradient check.
                    if i == 0 and not torch.equal(model(sample.to(reuse.device), use_cache=False).logits.view(torch.uint8),
                                                  reference.contiguous().view(torch.uint8)):
                        raise AssertionError("held-out prefix reuse changed complete-forward logits")
                    for name in result:
                        backups = {}
                        try:
                            if name == "head-mean":
                                for key, embedding in originals.items():
                                    backups[key] = embedding.embedding.weight.clone()
                                    embedding.embedding.weight.copy_(means[key])
                            elif name == "native-fp32":
                                for key, layer in model.model.engram_layers.items():
                                    layer.multi_head_embedding = native[key]
                            candidate = reuse.forward(sample, hidden, use_memory=name != "memory-off")
                            metrics = next_token_metrics(reference, candidate, sample.to(reuse.device))
                            metrics["window_index"] = i
                            if name == "native-fp32" and not metrics["logits_byte_exact"]:
                                raise AssertionError("native source logits changed")
                            result[name][split].append(metrics)
                            del candidate
                        finally:
                            for key, layer in model.model.engram_layers.items():
                                layer.multi_head_embedding = originals[key]
                                if key in backups:
                                    originals[key].embedding.weight.copy_(backups[key])
                    del reference
    finally:
        for embedding in native.values():
            embedding.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-model", "corpus", "upstream-dir", "protocol", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--native-library", type=Path)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--resume-from",type=Path,help="reuse final checkpoints and audited prefixes in an earlier run; preserve its report")
    p.add_argument("--prefix-from",type=Path,help="reuse audited prefixes only, with identical data and frozen-backbone computation")
    p.add_argument("--seed",type=int,help="run one preregistered seed as a parallel shard")
    p.add_argument("--smoke", action="store_true", help="training-only arithmetic/runtime check; no held-out scoring")
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["schema_version"] != 1 or args.output_dir.exists():
        p.error("use schema v1 and a fresh output directory")
    if args.resume_from and args.prefix_from or args.seed is not None and args.seed not in protocol["seeds"]:
        p.error("choose one prefix/resume source and a seed listed in the frozen protocol")
    if (protocol["training_windows"] % protocol["batch_size"] or protocol["epochs"] < 1
            or not 0 < protocol["training_fraction"] < protocol["test_start_fraction"] < 1):
        p.error("invalid training dimensions or contiguous split boundaries")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    report = {"schema_version": 1, "status": "running", "smoke_only": args.smoke,
              "command": shlex.join(sys.orig_argv), "code_sha": source_revision(),
              "protocol": protocol, "protocol_sha256": file_sha256(args.protocol),
              "upstream_sha": UPSTREAM_SHA,
              "source_file_hashes": {name: file_sha256(Path(__file__).parent / name) for name in
                                     ("train_tinyengram_identity.py", "frozen_tinyengram_prefix.py",
                                      "lm_quality_metrics.py", "tinyengram_checkpoint.py", "engram_row_adapter.py")},
              "environment": {"python": sys.version, "torch": torch.__version__,
                              "transformers": transformers.__version__, "numpy": np.__version__,
                              "os": platform.platform(), "device": args.device, "threads": protocol["threads"],
                              "interop_threads": 1}, "seeds": {},
              "limitations": ["finite late-layer branch training on one corpus; not convergence or released DeepSeek weights",
                              "sampled held-out windows; Qwen pretraining overlap unknown",
                              "same trained model inference ablations, not separately trained competing architectures",
                              "frozen-prefix training cache consumes disk/CPU RAM and has separate startup cost",
                              "quality oracle retains dense source memory; no serving RSS or latency claim",
                              "paired-window bootstrap excludes longer-range text dependence and seed-population uncertainty"]}
    started = time.perf_counter()

    def save():
        report["elapsed_seconds"] = time.perf_counter()-started
        write_json(report_path, report)

    try:
        if file_sha256(args.corpus) != protocol["corpus_sha256"]:
            raise ValueError("corpus differs from frozen protocol")
        previous = None
        prefix_root = args.resume_from or args.prefix_from
        if prefix_root:
            previous = json.loads((prefix_root/"report.json").read_text())
            match_fields = ("corpus_sha256","training_fraction","test_start_fraction","window_tokens",
                            "training_windows","batch_size","layer_from_end","threads")
            if (args.smoke or previous["smoke_only"] or previous["environment"]["device"] != args.device
                    or args.resume_from and previous["protocol"] != protocol
                    or any(previous["protocol"][k] != protocol[k] for k in match_fields)):
                raise ValueError("resume requires the identical non-smoke protocol and device")
            report["prefix_source"] = {"report_sha256":file_sha256(prefix_root/"report.json"),
                                "source_code_sha":previous["code_sha"],"previous_status":previous["status"],
                                "resume_final_checkpoints":bool(args.resume_from)}
        torch.set_num_threads(protocol["threads"])
        torch.set_num_interop_threads(1)
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but unavailable")
            device = torch.device("cuda",torch.cuda.current_device())
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            properties = torch.cuda.get_device_properties(device)
            report["environment"].update({"gpu":properties.name,"gpu_uuid":str(properties.uuid),
                        "gpu_total_memory":properties.total_memory,"cuda":torch.version.cuda,
                        "cudnn":torch.backends.cudnn.version(),"deterministic_algorithms":True,
                        "tf32":False})
            torch.cuda.reset_peak_memory_stats(device)
        try:
            report["environment"]["cpu"] = subprocess.check_output(["lscpu"], text=True)
        except (OSError, subprocess.CalledProcessError):
            report["environment"]["cpu"] = platform.processor()
        report["base_model_files"] = {f.name: {"bytes": f.stat().st_size, "sha256": file_sha256(f)}
                                      for f in args.base_model.iterdir() if f.is_file()}
        text = args.corpus.read_text()
        a, b = int(len(text)*protocol["training_fraction"]), int(len(text)*protocol["test_start_fraction"])
        boundaries = {"train": (0,a), "calibration": (a,b), "test": (b,len(text))}
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True, trust_remote_code=False)
        encoded = {s: tokenizer(text[begin:end], return_tensors="pt", add_special_tokens=False).input_ids
                   for s,(begin,end) in boundaries.items() if s == "train" or not args.smoke}
        report["splits"] = {s: {"character_range": boundaries[s], "tokens": ids.shape[1],
                                 "token_ids_sha256": tensor_sha(ids)} for s,ids in encoded.items()}
        length, batch = protocol["window_tokens"], protocol["batch_size"]
        training, selected = sample_windows(encoded["train"], length, protocol["training_windows"])
        if args.smoke:
            training, selected = training[:batch], selected[:batch]
        train_targets = len(training)*(length-1)*(1 if args.smoke else protocol["epochs"])
        report["training_targets_per_seed"] = train_targets
        report["sampled_train_windows"] = {"count": len(training), "token_ids_sha256": tensor_sha(training),
                                           "window_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest()}
        evaluation = {s: ids[0,:length*protocol[s+"_windows"]].reshape(-1,length)
                      for s,ids in encoded.items() if s != "train"}
        for s,ids in evaluation.items():
            if len(ids) != protocol[s+"_windows"]:
                raise ValueError("insufficient held-out windows")
        config = Qwen3Config.from_pretrained(args.base_model, local_files_only=True)
        layer_id = config.num_hidden_layers-protocol["layer_from_end"]
        if layer_id < 1:
            raise ValueError("backbone is too short for a nonempty frozen prefix")
        cache_files, cache_metadata = {}, None
        if previous is not None:
            cache_metadata = previous["prefix_preparation"]
            report["prefix_preparation"] = cache_metadata
            report["prefix_files"] = previous["prefix_files"]
            for split,metadata in previous["prefix_files"].items():
                path = prefix_root/(split+"-prefix.npy")
                if file_sha256(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
                    raise ValueError("resume prefix checksum changed: "+split)
                cache_files[split] = np.load(path,mmap_mode="r",allow_pickle=False)
            if previous["sampled_train_windows"] != report["sampled_train_windows"] or previous["splits"] != report["splits"]:
                raise ValueError("resume data or sampling schedule changed")
        seeds_to_run = [args.seed] if args.seed is not None else protocol["seeds"][:1] if args.smoke else protocol["seeds"]
        report["executed_seeds"] = seeds_to_run
        for seed in seeds_to_run:
            torch.manual_seed(seed)
            engram, qwen = load_upstream(args.upstream_dir)
            qwen.set_skip_engram(False)
            ecfg = engram.EngramConfig(tokenizer_name_or_path=str(args.base_model.resolve()),
                         engram_vocab_size=protocol["ngram_vocab_sizes"], max_ngram_size=protocol["max_ngram_size"],
                         n_embed_per_ngram=protocol["n_embed_per_ngram"], n_head_per_ngram=protocol["n_head_per_ngram"],
                         layer_ids=[layer_id], pad_id=tokenizer.pad_token_id, kernel_size=protocol["kernel_size"], seed=seed)
            engram.set_global_step(0)
            prior_seed = previous["seeds"].get(str(seed)) if args.resume_from else None
            resuming = bool(prior_seed and prior_seed.get("steps") == protocol["epochs"]*len(training)//batch
                            and (args.resume_from/("seed-"+str(seed))/"checkpoint").is_dir())
            model_path = args.resume_from/("seed-"+str(seed))/"checkpoint" if resuming else args.base_model
            model, loading = qwen.EngramQwenForCausalLM.from_pretrained(model_path, engram_cfg=ecfg,
                         local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager",
                         output_loading_info=True)
            expected_missing = set() if resuming else {name for name in model.state_dict() if "engram_layers" in name}
            if (set(loading["missing_keys"]) != expected_missing or loading["unexpected_keys"]
                    or loading["mismatched_keys"] or loading["error_msgs"]):
                raise ValueError("base backbone was not exactly restored: " + repr(loading))
            for name, parameter in model.named_parameters():
                parameter.requires_grad_("engram_layers" in name)
            model.to(device).eval()
            reuse = FrozenTinyEngramPrefix(model)
            frozen_hash = reuse.frozen_weight_sha256()
            seed_report = report["seeds"][str(seed)] = {"status": "training", "layer_id": layer_id,
                   "frozen_weight_sha256": frozen_hash, "losses": [], "initial_loading": loading,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "reuse_check": check_reuse(model, reuse, training[:batch], gradients=True)}
            branch = reuse.branch
            weight = branch.multi_head_embedding.embedding.weight
            initial = weight.detach().clone()
            seen = np.zeros(len(weight),dtype=bool)
            seed_report["memory"] = {"rows": len(weight), "dim": weight.shape[1], "raw_bytes": weight.numel()*4,
                                     "initial_weight_sha256": tensor_sha(initial),
                                     "head_offsets": branch.multi_head_embedding.offsets.tolist()}
            if resuming:
                proof = seed_report["reuse_check"]
                if tensor_sha(weight) != prior_seed["memory"]["trained_weight_sha256"]:
                    raise AssertionError("resumed source memory weights changed")
                seed_report = report["seeds"][str(seed)] = json.loads(json.dumps(prior_seed))
                seed_report.update({"status":"resumed_evaluation","resumed_steps":prior_seed["steps"],
                                    "resume_reuse_check":proof,"resume_loading":loading,
                                    "resume_checkpoint_sha256":file_sha256(model_path/"model.safetensors")})
            if cache_metadata is None:
                prep_start = time.perf_counter()
                for split,windows in {"train": training, **evaluation}.items():
                    path = args.output_dir / (split+"-prefix.npy")
                    states = np.lib.format.open_memmap(path, mode="w+", dtype="<f4",
                                    shape=(len(windows),length,config.hidden_size))
                    # Evaluation prefix batches match its batch-one forward.
                    prep_batch = batch if split == "train" else 1
                    for i in range(0,len(windows),prep_batch):
                        states[i:i+prep_batch] = reuse.capture(windows[i:i+prep_batch]).cpu().numpy()
                        if i % 64 == 0:
                            print(f"prefix {split} {i}/{len(windows)}",flush=True)
                    states.flush()
                    states._mmap.close()
                    cache_files[split] = np.load(path,mmap_mode="r",allow_pickle=False)
                    report.setdefault("prefix_files",{})[split] = {"bytes":path.stat().st_size,
                                                 "sha256":file_sha256(path)}
                cache_metadata = {"frozen_weight_sha256":frozen_hash,"seconds":time.perf_counter()-prep_start}
                report["prefix_preparation"] = cache_metadata
                save()
            elif cache_metadata["frozen_weight_sha256"] != frozen_hash:
                raise AssertionError("seed changed the frozen backbone; prefix cache is invalid")
            # Check the serialized state and sample schedule as well as the helper.
            serialized = torch.from_numpy(np.array(cache_files["train"][:batch],copy=True)).to(device)
            with torch.no_grad():
                if not torch.equal(model(training[:batch].to(device),use_cache=False).logits.view(torch.uint8),
                                   reuse.forward(training[:batch],serialized).contiguous().view(torch.uint8)):
                    raise AssertionError("serialized prefix cache changed source logits")
            for windows in training.split(64):
                queried = torch.from_numpy(branch.hash_mapping.hash(windows.numpy())[layer_id])
                shifted = queried + branch.multi_head_embedding.offsets.cpu()
                seen[shifted.flatten().numpy()] = True
            params = [p for p in model.parameters() if p.requires_grad]
            other = [p for p in params if p is not weight]
            optimizer = torch.optim.Adam([{"params":[weight],"lr":protocol["memory_learning_rate"]},
                                          {"params":other,"lr":protocol["branch_learning_rate"]}])
            rng = np.random.default_rng(seed)
            step = prior_seed["steps"] if resuming else 0
            train_start = time.perf_counter()
            for epoch in range(0 if resuming else 1 if args.smoke else protocol["epochs"]):
                order = rng.permutation(len(training))
                for begin in range(0,len(training),batch):
                    indices = order[begin:begin+batch]
                    samples = training[torch.from_numpy(indices)]
                    states = torch.from_numpy(np.array(cache_files["train"][indices],copy=True)).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    engram.set_global_step(step)
                    logits = reuse.forward(samples,states)
                    loss = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),samples[:,1:].to(device).reshape(-1))
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params,protocol["gradient_clip"])
                    optimizer.step()
                    seed_report["losses"].append(float(loss.detach()))
                    step += 1
                    del logits,loss
                    if step % 32 == 0 or args.smoke:
                        seed_report.update({"step":step,"training_seconds":time.perf_counter()-train_start})
                        save()
                        print(f"seed {seed} epoch {epoch+1} step {step} loss={seed_report['losses'][-1]:.6f}",flush=True)
            if reuse.frozen_weight_sha256() != frozen_hash:
                raise AssertionError("training changed frozen weights")
            seed_report.update({"steps":step,"training_seconds":prior_seed["training_seconds"] if resuming else time.perf_counter()-train_start,
                                "frozen_weight_hash_unchanged":True})
            if device.type == "cuda":
                seed_report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                seed_report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
            if not resuming:
                seed_report["memory"].update({"queried_rows":int(seen.sum()),
                              "changed_rows":int((weight.detach()!=initial).any(1).sum()),
                              "trained_weight_sha256":tensor_sha(weight)})
            if args.smoke:
                seed_report["status"] = "smoke_passed"
                del optimizer,reuse,model
                continue
            directory = args.output_dir / ("seed-"+str(seed))
            directory.mkdir()
            checkpoint,bundle = directory/"checkpoint",directory/"bundle"
            model.save_pretrained(checkpoint,safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            write_json(checkpoint/"engram_config.json",ecfg.to_dict())
            write_json(checkpoint/"trainer_state.json",{"global_step":step,"training_prefix_characters":a,
                                                     "corpus_sha256":protocol["corpus_sha256"]})
            engram.set_global_step(step)
            exported = export_tinyengram_checkpoint(checkpoint,bundle,upstream_dir=args.upstream_dir,
                                                    block_rows=64,predictor="xor")
            windows = evaluate_seed(model,reuse,evaluation,cache_files,bundle,args.native_library)
            summaries = {s:{name:summarize_windows(v[s]) for name,v in windows.items()} for s in evaluation}
            quality = {"seed":seed,"windows":windows,"summary":summaries,
                       "memory_qualified_on_calibration":memory_qualification(summaries["calibration"],minimum_ppl_effect=protocol["minimum_ppl_effect"]),
                       "test_confirmation":memory_qualification(summaries["test"],minimum_ppl_effect=protocol["minimum_ppl_effect"])}
            write_json(directory/"quality.json",quality)
            seed_report.update({"status":"passed","quality":{k:v for k,v in quality.items() if k!="windows"},
                                "quality_sha256":file_sha256(directory/"quality.json"),
                                "bundle_manifest_sha256":file_sha256(bundle/"bundle.json"),
                                "complete_checkpoint_bytes":sum(f.stat().st_size for f in checkpoint.rglob('*') if f.is_file()),
                                "complete_bundle_bytes":sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file())})
            save()
            del optimizer,params,other,weight,branch,reuse,model,initial
        report["status"] = "smoke_passed" if args.smoke else "passed"
        if not args.smoke:
            report["all_seeds_qualified_on_calibration"] = set(seeds_to_run)==set(protocol["seeds"]) and all(r["quality"]["memory_qualified_on_calibration"] for r in report["seeds"].values())
            report["all_seeds_confirmed_on_test"] = set(seeds_to_run)==set(protocol["seeds"]) and all(r["quality"]["test_confirmation"] for r in report["seeds"].values())
    except BaseException as error:
        report["status"],report["error"] = "failed",repr(error)
        raise
    finally:
        save()
        print("Report:",report_path.resolve(),flush=True)


if __name__ == "__main__":
    main()
