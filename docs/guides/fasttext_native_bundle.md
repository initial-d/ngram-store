# FastText: native storage for me_fasttext models

A dense me_fasttext `.bin` model can be converted into a native bundle. The
converted model loads with the same `loadModel(path)` call and answers the same
queries, using the original dictionary and row IDs; the input matrix is stored in
compressed native blocks behind a DATrie key index. On a real 90,380-row model,
word and sentence vectors stay byte-identical, the bundle is 0.46% smaller, and
hot queries are slower than the dense model
([report](../../evidence/fasttext_native_evidence_20261031/README.md)).

Linux x86-64 only: the trainer and the bundled DATrie libraries are built for it.

## Train, convert and query a sample

This uses the small text in `examples/` and needs no downloads or GPU. It checks
that the pipeline works; the sample is too small to say anything about
compression.

```bash
python -m pip install .
make fasttext-native native structural-native
mkdir -p artifacts/fasttext-sample
ngram-store prepare-fasttext --input examples/fasttext_sample.txt \
  --output artifacts/fasttext-sample/pairs.txt
```

The trainer writes legacy trie dumps into the working directory, so run it from
the output directory. Flags use a single dash:

```bash
cd artifacts/fasttext-sample
../../build/fasttext-native skipgram -input pairs.txt -output model \
  -dim 32 -minCount 1 -minn 2 -maxn 6 -epoch 1 -thread 1 -verbose 0
cd ../..
ngram-store fasttext-export --model artifacts/fasttext-sample/model.bin \
  --output-dir artifacts/fasttext-sample/model.native \
  --fasttext-binary build/fasttext-native \
  --structural-library build/libmeft_structural.so
printf 'memory\nvectors\nlookup\n' | build/fasttext-native print-word-vectors \
  artifacts/fasttext-sample/model.native
```

`prepare-fasttext` turns each line into `p#0 query\x01source` pairs, keeping word
order. This custom trainer does not read plain whitespace text like upstream
fastText does.

## Convert your own model

```bash
ngram-store fasttext-export \
  --model model.bin --output-dir model.native \
  --fasttext-binary build/fasttext-native \
  --structural-library build/libmeft_structural.so \
  --mode exact

printf 'example\nexamples\n' | build/fasttext-native print-word-vectors model.native
printf 'an example sentence\n' | build/fasttext-native print-sentence-vectors model.native
```

Every command accepts either the `.bin` or the bundle directory. For supervised
models, `predict`, `predict-prob` and `test` work on bundles too. `nn` and
`analogies` still build all word vectors in memory. Keep the original `.bin`: it is
needed for further training, `quantize`, `saveModel` and `getInputMatrix`, which
refuse native input.

The converter is also available as `meft-fasttext-export` and
`ngram_store.export_fasttext_model`, without a PyTorch dependency.

From C++, query calls are unchanged; build with `-DMEFT_NATIVE_FASTTEXT` and link
`native/fasttext_bundle.cc` and the C reader, as the Make target does, then
recompile your callers:

```cpp
fasttext::FastText model;
model.loadModel("model.native");
fasttext::Vector word(model.getDimension());
model.getWordVector(word, "example");
```

## Modes and behaviour

- `layout` keeps every row; `exact` also shares byte-identical prefix/suffix rows.
  Both check the SHA-256 of the full reconstructed input matrix, untrained rows
  included. `joint` merges similar rows (cosine and norm ratio) and changes
  vectors; its manifest records both source and reconstructed hashes.
- Queries look up original IDs in `keys.das`, map them to physical rows, and add
  rows in the original order in float32, so results match the dense model.
- Defaults: 8,192-byte decoded blocks, 4 MiB block cache, at most 256 rows per
  gather. Change with `--max-block-bytes`, `--cache-bytes`, `--query-chunk-rows`.
- Unknown subwords are skipped instead of reading past the matrix, missing keys
  return `-1` or a zero vector, and empty sentences return zero rather than NaN.
  These were undefined in the old code; equivalence holds for valid queries.
- The dictionary, key index and dense output matrix stay in memory, as does an
  unused 200 MB word hash inherited from the old loader. Touched mmap pages count
  as file-backed memory.

The bundle holds `table.men`, `keys.das`, `model.meta` (args, dictionary, output
matrix), reader settings and audit manifests. Only load bundles you produced: the
legacy metadata loader is not hardened against malicious input, and the C++ facade
does not checksum the whole bundle when opening it.

Supported input: this repository's dense, unquantized, unpruned models. Official
fastText `.bin`, quantized `.ftz` and old `.z` files are different formats; `.z`
keeps its own FTIndex reader ([compatibility](legacy_fasttext.md)).
