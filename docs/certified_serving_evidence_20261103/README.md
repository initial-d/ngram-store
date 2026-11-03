# Complete native build and original-ID certified serving

On 2026-11-03, two fresh single-threaded CPU workers produce the same ready-to-query
artifact from the existing 1,855,907 × 256 FP32 trained table and unchanged
142,308-bag domain. Neither worker opens native C/runtime state. A third process
performs native and actual PyTorch CPU verification, with its RSS recorded separately.

| Complete build | Peak RSS MiB | Wall seconds | Complete serving bytes |
| --- | ---: | ---: | ---: |
| Legacy fit/means + expanded staging + marked raw native pack | 3,143.5 | 31.53 | 399,867,577 |
| External fit/means + direct raw native export / 64 MiB managed buffers | 100.7 | 129.53 | 399,868,858 |

The new complete path uses **31.21× less peak process RSS (96.80% reduction)**
and takes **4.11× longer**. All **385,017,776 native table bytes** match both
the fresh legacy control and the earlier raw certified artifact, SHA256
`2e6131f707097ebcf9377338b31d0247c53f25974b754da168569821638e89e7`.
The partition still has 360,088 groups and the same mapping/domain hashes.
The 1,281-byte serving-size difference is manifest reporting overhead, not
different vectors or a new compression improvement.

The external native exporter writes one descriptor at a time and copies dense
group IDs into the original-ID forwarding map, avoiding resident descriptor/
representative dictionaries and the legacy 1,900,448,768-byte expanded staging
file. The final serving directory contains only `table.men`, `domain.npz` and
`manifest.json`. Raw blocks have 8 rows. Complete backend/certificate/manifest
bytes are counted; application keys, model output/word weights and source CSR
remain separate. This is not whole-model memory accounting.

External stage-sampled workspace is recorded in [external.json](external.json);
the float64 accumulator is 737,460,224 logical bytes. Temporary disk still
scales with input/groups and managed buffers **do not cap RSS or OS page cache**.
The earlier [33.06× fit/means result](../external_incidence_evidence_20261102/README.md)
is a different, narrower experiment. The complete-path numbers above are fresh
measurements, not extrapolation from that result. The legacy and external final
files are compared byte by byte after workers exit.

The separate [verifier](verifier.json) loads both certified native artifacts,
checks the entire file hash, compares the legacy reader and every plan array on
the unchanged 512-bag/21,747-occurrence trace, and verifies native sum/mean output
bytes. It also rejects an incomplete multi-member group. Native output hashes
match the previous study. The frozen `CertifiedEmbeddingBag` CPU module returns
those same native bytes for the actual trace.

Against source FP32 PyTorch `nn.EmbeddingBag`, mean-output maximum absolute
difference is **4.77e-7**, global relative L2 **7.76e-8**, with different bytes.
Projection changes rows and the native reducer uses FP64; real-arithmetic
certification is distinct from source FP32 reduction order/rounding. This is an
operator integration check, not downstream or LLM-quality evidence.

The verifier takes 4.82 seconds and peaks at **4,280.5 MiB RSS**: it deliberately
loads resident maps/native tables and copies the full source matrix for the
PyTorch control. This cost is excluded from **both** construction workers and
must not inherit their buffer claim. Runtime map loading and query compilation
remain resident, single-worker operations. No serving-latency claim is made.

The included offline NumPy example actually trains SGNS on MIT repository-authored
text: 919 rows, 1,200 updates and 47 checked queries. Every native canonical row
matches the legacy projection, all rows receive a training change and an incomplete
query is refused. Its [demo report](demo-report.json) is an integration fixture,
not representative compression, semantic quality, convergence or performance
evidence. Public-corpus text, CSR and large trained weights are not redistributed.

Remote full suite: **184 tests, 17 skips**; targeted new bundle/PyTorch tests:
**4 tests, no skips**. Local Python 3.12 NumPy-only suite: **184 tests, 100 skips**.
Skip counts denote unavailable optional paths, not passed integration tests.
Linux/macOS installed wheel checks and the outside-checkout standalone runner
run in CI. [Raw tests](tests/), [environment](environment.json),
[raw build report](report.json), [actual commands](reproduction/run.sh).
The exact Linux wheel from source `49a0bfe` also passes the standalone runner
with the actual trained example and PyTorch SGNS head on the user's remote
host, outside the checkout: [installed reproduction](installed-linux-reproduction.json).
Check numeric/provenance requirements and regenerate file hashes with
`python docs/certified_serving_evidence_20261103/verify.py`.

Build/verifier source:
[3f0770c](https://github.com/initial-d/ngram-store/commit/3f0770cd99b08b3d0465ff26d9d474308124fdd1),
with source/input/trace/native hashes in the reports. AMD EPYC 7742, Linux
5.4/glibc 2.35, Python 3.11.15, NumPy 2.2.6, PyTorch 2.11.0+cu130 for the CPU
verifier, Ceph filesystem and GCC 11.4 native reader. CPU/filesystem are shared
and warm; no exclusive/cold-cache, GPU, monetary or energy result. No GPU was
inspected or manipulated. No build, verification or targeted-test failures.
[Predeclared protocol](../certified_serving_protocol_20261103.md),
[try the released flow](../try_certified_memory.md).

An installed example and reproducible entry improve usability. They do not
establish independent third-party reproduction, adoption, stars or citations.
