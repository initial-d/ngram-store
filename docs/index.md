# NGramStore documentation

## Start here

- [Try a trained table in minutes](guides/try_certified_memory.md): install, offline demo, your own table
- [Engine guide: pack, inspect, query](guides/evaluator_quickstart.md)
- [C ABI, native format and runtime boundaries](guides/native_ngram_memory.md)

## Guides

**Engine features**

- [Certified co-training groups](guides/cotraining_groups.md): shared parameters for rows with identical bag incidence
- [Partial-group queries with L2 error budgets](guides/budgeted_bag_queries.md)
- [Disk-backed incidence construction](guides/external_incidence_build.md)
- [Host memory → device gather and prefetch](guides/device_memory_prefetch.md)

**Applications**

- [FastText: prepare, train, export, query](guides/fasttext_native_bundle.md)
- [Engram LLM memory: official demo and factory](guides/engram_quickstart.md)
- [Engram-style storage and lookup](guides/engram_storage.md) · [PyTorch gather adapter](guides/engram_gather_adapter.md)
- [CPU news classification](guides/news_classifier_app.md) · [open-vocabulary hybrids](guides/news_shift_app.md)

**Compatibility and citation**

- [me_fasttext compatibility](guides/legacy_fasttext.md) · [legacy export contract](guides/legacy_merge_contract.md)
- [Citation guide](guides/citation_guide.md)

## Roadmap

- [Engine roadmap, priorities and acceptance gates](roadmap.md)

## Evidence

All experiment protocols, results and verifiers are in the
[evidence archive](../evidence/README.md), grouped by research question.

## History

[me_fasttext-era documents](history/README.md): the original prototype, its
engineering notes and early benchmark material, kept for reference.
