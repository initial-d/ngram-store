"""Paired next-token metrics; retain scalars rather than corpus-sized logits.

Window bootstrap intervals describe this fixed text/checkpoint, not independent
documents or a population of trained models. Heavy dependencies are optional.
"""
import math


def next_token_metrics(source, candidate, ids):
    """Score positions [0,L-1) against ids[:,1:]; KL is source || candidate.

    Both forwards must use the same execution schedule. Float64 normalization
    avoids reporting negative KL from FP32 cancellation for close candidates.
    The final position has no observed target and is excluded from all metrics.
    """
    import torch
    if (source.ndim != 3 or source.shape != candidate.shape or ids.ndim != 2
            or source.shape[:2] != ids.shape or ids.shape[1] < 2
            or source.device != candidate.device or source.device != ids.device
            or ids.dtype not in (torch.int32, torch.int64)
            or not torch.isfinite(source).all() or not torch.isfinite(candidate).all()):
        raise ValueError("matching finite [batch,length,vocab] logits and integer IDs required")
    labels = ids[:, 1:].long()
    if labels.min() < 0 or labels.max() >= source.shape[-1]:
        raise ValueError("target IDs outside vocabulary")
    a = source[:, :-1].double().log_softmax(-1)
    b = candidate[:, :-1].double().log_softmax(-1)
    nll_a = -a.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    nll_b = -b.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    kl = (a.exp() * (a - b)).sum(-1)
    if float(kl.min()) < -1e-10:
        raise ArithmeticError("negative KL beyond rounding tolerance")
    return {
        "targets": labels.numel(),
        "source_nll": nll_a.flatten().tolist(),
        "candidate_nll": nll_b.flatten().tolist(),
        "source_to_candidate_kl": kl.clamp_min(0).flatten().tolist(),
        "top1_changes": int((a.argmax(-1) != b.argmax(-1)).sum()),
        "source_correct": int((a.argmax(-1) == labels).sum()),
        "candidate_correct": int((b.argmax(-1) == labels).sum()),
        "logits_byte_exact": torch.equal(source.contiguous().view(torch.uint8),
                                         candidate.contiguous().view(torch.uint8)),
        "max_absolute_logit_delta": float((source - candidate).abs().max()),
    }


def paired_window_interval(windows, *, seed=731, resamples=10000):
    """Resample paired windows; weight each replicate by its target count."""
    import numpy as np
    if not windows or resamples < 1:
        raise ValueError("nonempty windows and positive bootstrap count required")
    counts = np.array([w["targets"] for w in windows], dtype=np.float64)
    differences = np.array([math.fsum(w["candidate_nll"]) - math.fsum(w["source_nll"])
                            for w in windows])
    if np.any(counts <= 0):
        raise ValueError("positive target counts required")
    draws = np.random.default_rng(seed).integers(len(windows), size=(resamples, len(windows)))
    values = differences[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.quantile(values, [.025, .975])
    return {"delta_nll_95ci": [float(lo), float(hi)],
            "relative_ppl_change_95ci": [math.expm1(float(lo)), math.expm1(float(hi))],
            "seed": seed, "resamples": resamples, "unit": "paired nonoverlapping token window"}


def summarize_windows(windows):
    import numpy as np
    if not windows:
        raise ValueError("nonempty windows required")
    n = sum(w["targets"] for w in windows)
    a = math.fsum(x for w in windows for x in w["source_nll"]) / n
    b = math.fsum(x for w in windows for x in w["candidate_nll"]) / n
    kl = np.array([x for w in windows for x in w["source_to_candidate_kl"]])
    return {"windows": len(windows), "targets": n, "source_nll": a, "candidate_nll": b,
            "source_ppl": math.exp(a), "candidate_ppl": math.exp(b),
            "delta_nll": b - a, "relative_ppl_change": math.expm1(b - a),
            "mean_kl": float(kl.mean()), "p99_kl": float(np.quantile(kl, .99)),
            "max_kl": float(kl.max()), "top1_changes": sum(w["top1_changes"] for w in windows),
            "source_correct": sum(w["source_correct"] for w in windows),
            "candidate_correct": sum(w["candidate_correct"] for w in windows),
            "all_logits_byte_exact": all(w["logits_byte_exact"] for w in windows),
            "max_absolute_logit_delta": max(w["max_absolute_logit_delta"] for w in windows),
            "bootstrap": paired_window_interval(windows)}


def memory_qualification(summary, *, minimum_ppl_effect=.01):
    """Require both the whole branch and row identity to help on calibration."""
    return all(summary[name]["relative_ppl_change"] >= minimum_ppl_effect
               and summary[name]["bootstrap"]["delta_nll_95ci"][0] > 0
               for name in ("memory-off", "head-mean"))
