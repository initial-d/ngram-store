"""Dependency-free audit of fixed windows, source pairing and reported metrics.

Bootstrap generation is reproducible with the archived NumPy evaluator and its
seed. This audit recomputes deterministic metrics and gate decisions from the
retained per-target scalars; it never needs corpus text or model weights.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    with gzip.open(ROOT / "qualification.json.gz", "rt") as handle:
        report = json.load(handle)
    summary_file = json.loads((ROOT / "summary.json").read_text())
    for key, value in summary_file.items():
        assert report[key] == value, key
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert report["protocol"] == protocol
    assert report["protocol_sha256"] == hashlib.sha256((ROOT / "protocol.json").read_bytes()).hexdigest()
    assert protocol["excluded_prefix_characters"] == 16384
    assert protocol["window_tokens"] == 128
    assert (protocol["calibration_windows"], protocol["test_windows"]) == (16, 48)
    assert protocol["minimum_ppl_effect"] == .01
    assert report["status"] == "passed"
    assert report["code_sha"] == manifest["runtime_commit"]
    assert report["environment"]["device"] == "cpu"
    assert report["environment"]["threads"] == 8
    test_runs = json.loads((ROOT / "isolated-tests" / "summary.json").read_text())["results"]
    assert len(test_runs) == 33 and all(run["exit_code"] == 0 for run in test_runs)
    integration_log = (ROOT / "integration-tests.log").read_text()
    assert "test_complete_quality_oracle_on_trained_fixture" in integration_log
    assert "Ran 19 tests" in integration_log and "OK (skipped=3)" in integration_log
    for name, digest in report["source_file_hashes"].items():
        assert hashlib.sha256((ROOT / "runtime" / name).read_bytes()).hexdigest() == digest, name
    assert set(report["windows"]) == {"memory-off", "head-mean", "native-fp32"}
    expected_counts = {"calibration": (16, 2032, 0), "test": (48, 6096, 16)}
    for split, (count, targets, start) in expected_counts.items():
        sources = report["windows"]["memory-off"][split]
        for name, data in report["windows"].items():
            windows = data[split]
            assert len(windows) == count
            assert [w["window_index"] for w in windows] == list(range(start, start + count))
            for i, w in enumerate(windows):
                assert w["targets"] == 127
                assert w["source_nll"] == sources[i]["source_nll"]
                for key in ("source_nll", "candidate_nll", "source_to_candidate_kl"):
                    assert len(w[key]) == 127 and all(math.isfinite(x) and x >= 0 for x in w[key])
                for key in ("top1_changes", "source_correct", "candidate_correct"):
                    assert 0 <= w[key] <= 127
                if name == "native-fp32":
                    assert w["logits_byte_exact"] and w["max_absolute_logit_delta"] == 0
                    assert w["candidate_nll"] == w["source_nll"]
                    assert w["source_to_candidate_kl"] == [0.] * 127
            summary = report["summary"][split][name]
            assert (summary["windows"], summary["targets"]) == (count, targets)
            a = math.fsum(x for w in windows for x in w["source_nll"]) / targets
            b = math.fsum(x for w in windows for x in w["candidate_nll"]) / targets
            kl = [x for w in windows for x in w["source_to_candidate_kl"]]
            for key, value in {"source_nll": a, "candidate_nll": b, "delta_nll": b-a,
                               "source_ppl": math.exp(a), "candidate_ppl": math.exp(b),
                               "relative_ppl_change": math.expm1(b-a), "mean_kl": math.fsum(kl)/targets,
                               "p99_kl": quantile(kl, .99), "max_kl": max(kl)}.items():
                close(summary[key], value)
            for key in ("top1_changes", "source_correct", "candidate_correct"):
                assert summary[key] == sum(w[key] for w in windows)
            assert summary["all_logits_byte_exact"] == all(w["logits_byte_exact"] for w in windows)
            bootstrap = summary["bootstrap"]
            assert (bootstrap["seed"], bootstrap["resamples"]) == (731, 10000)
            lo, hi = bootstrap["delta_nll_95ci"]
            assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
            for x, y in zip(bootstrap["relative_ppl_change_95ci"], (math.expm1(lo), math.expm1(hi))):
                close(x, y)
    for split, key in (("calibration", "memory_qualified_on_calibration"), ("test", "test_confirmation")):
        qualified = all(report["summary"][split][name]["relative_ppl_change"] >= .01
                        and report["summary"][split][name]["bootstrap"]["delta_nll_95ci"][0] > 0
                        for name in ("memory-off", "head-mean"))
        assert report[key] == qualified
    print("Verified 8,128 paired targets, native output identity, fixed protocol and qualification decisions.")


if __name__ == "__main__":
    main()
