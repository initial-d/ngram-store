# NGramStore documentation

## Start here

- [Concepts](concepts.md): the terms used across guides and reports
- [Engine guide](guides/evaluator_quickstart.md): pack, inspect and query a table
- [C ABI and runtime reference](guides/native_ngram_memory.md): format, cache, PyTorch and bag adapters

## Guides

**Engine features**

- [Certified bag tables](guides/try_certified_memory.md): shared group means for sum/mean models, with a runnable demo
- [Certified co-training groups](guides/cotraining_groups.md): the math behind them
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

## Technical report

[Deduplicate, Don't Compress: Storing and Serving Learned N-gram Memory](report/ngramstore-report.pdf)
([source and build](report/README.md))

## Roadmap

- [Engine roadmap, priorities and acceptance gates](roadmap.md)

## Evidence

All experiment protocols, results and verifiers are in the
[evidence archive](../evidence/README.md), grouped by research question.

## History

[me_fasttext-era documents](history/README.md): the original prototype, its
engineering notes and early benchmark material, kept for reference.
