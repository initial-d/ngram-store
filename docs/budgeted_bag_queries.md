# Partial-group queries with an explicit L2 budget

The default certified path rejects a query unless every accessed group's members
have equal counts. The optional residual-bound path also admits partial or unequal
groups when the **actual-source error upper bound** fits a user-supplied budget.
It uses trained source vectors, with no premise that all same-prefix/suffix vectors
have high cosine. It is an approximation contract, separate from lossless packing
and the weight-independent certificate.

## Source checkout API and CLI

This addition is in source after v0.4.0; the v0.4.0 release wheels do not contain it.
Install a checkout with `python -m pip install '.[incidence]'`.

First build the existing native compact table, then a separate residual sidecar:

```bash
ngram-store compact-bags --native --input vectors.f32 --dim 256 \
  --bag-ids ids.u64 --bag-offsets offsets.u64 --output memory

ngram-store build-error-bounds --input vectors.f32 --store memory --output bounds

ngram-store query-bags --store memory --bounds bounds --ids 0 1 0 \
  --mode mean --max-relative-l2 0.01
```

The last command permits at most 1% **source-relative aggregate L2 error**, not
99% cosine or a 1% downstream-loss change. Both supplied absolute/relative budgets
must pass; any failed bag refuses the entire batch. Querying without `--bounds`
and a budget still uses the original strict certificate.

```python
from ngram_store import build_error_bounds, BudgetedBagRows

build_error_bounds("vectors.f32", "memory", "bounds")
with BudgetedBagRows("memory", "bounds") as table:
    # assess exposes candidate values/bounds for diagnostics, without accepting
    # them. aggregate applies the explicit budget and refuses failed batches.
    candidate = table.assess([[0, 1, 0]], mean=True)
    result = table.aggregate([[0, 1, 0]], mean=True, max_relative_l2=0.01)
    vectors = result.vectors
    report = result.summary()
```

IDs above are illustrative; the real source and query determine acceptance.
Source matrix, native table and partition hashes bind the sidecar. Array hashes
are verified on open. The original source matrix is not needed at query time.
Artifacts/handles retain the trusted-immutable, single-worker boundary.

## Bound and cancellation

Let `m_g` be the FP32 row actually decoded for group `g`, `e_i` the declared
FP32 source and `r_i = e_i - m_g` in real arithmetic. Store conservative scalars:

```
R_i >= ||r_i||_2
S_g >= sum_{i in g} R_i
D_g >= ||sum_{i in g} r_i||_2
```

For query member counts `c_i` and any nonnegative centering count `t`:

```
sum_i c_i r_i = sum_i (c_i - t) r_i + t sum_i r_i
||sum_i c_i r_i||_2 <= sum_i |c_i - t| R_i + t D_g
```

The convex piecewise-linear bound attains a minimum at zero or an observed
positive count. Missing members have count zero. Their total residual mass is
bounded by `S_g` minus an outward-rounded lower sum of observed `R_i` values;
runtime does not scan all members of a partially queried group. Centering can
exploit cancellation even for queries outside the strict domain. `t=0` supplies
the ordinary triangle bound, so minimizing also retains that option.

Rounded FP32 means can have a nonzero group residual sum. The sidecar measures
and bounds that defect rather than assuming `D_g=0`. Source subtraction, norm
evaluation, accumulation and bound arithmetic use outward rounding and IEEE
round-to-nearest error factors. Native binary64 reduction and its binary32 output
cast are separately bounded. Mean mode additionally includes division/output
rounding. Source comparison is a real-arithmetic sum/mean of declared FP32
values; it is not byte identity with PyTorch's FP32 reduction.

For returned vector `y` and total absolute bound `B`, the source norm is at least
`||y||_2 - B`. A positive conservative lower bound gives:

```
relative L2 error <= B / (||y||_2 - B)
```

If that denominator cannot be proved positive, the relative bound is unbounded
and relative-budget queries are refused. An absolute budget can still be useful.
Empty bags have zero output and zero bound. Nonfinite outputs are refused.
Exact-zero budgets can be conservative because numerical rounding is included;
use the strict weight-independent path when that is the desired contract.

## Complete costs and limits

Sidecar array bytes are `8*N + 16*G`, plus its JSON metadata. The optional build
also uses two temporary `G*d*8` files, native decoding, O(N+G) runtime maps and
chunked source work. Scratch/RSS/time costs belong to this new step and do not
inherit the earlier 31.21x complete-builder result or a hard memory cap. Arrays
are read-only mmap at runtime; query temporary storage scales with occurrences.

Bounds may be loose, especially with cancellation between groups. A rejected
query can still have low measured error. Above-budget queries need an explicitly
chosen dense fallback or rejection; this API does not retain a source fallback.
It does not prove semantic/downstream quality, an Engram gate contract, universal
OOV coverage, quantization dominance, or a novel general error-bound theorem.
The contribution to evaluate is incidence-aware, source-specific coverage at
complete storage and query cost. [Frozen-checkpoint protocol](budgeted_bag_protocol_20261104.md).
