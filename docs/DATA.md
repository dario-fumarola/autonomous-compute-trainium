# Data policy

Use existing natural source documents, without generated arithmetic, model-written rewrites, teacher solutions, or newly generated training examples. FineWeb-Edu and FineMath use model-assisted *selection* of web documents; the source text is not a generated teaching trace. Web corpora can still contain unidentified machine-written pages; do not claim that every document is human-authored. The initial FineWeb shard is from the 2013 crawl. CodeParrot is an older GitHub collection.

Initial qualification sources:

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), natural educational web text.
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), mathematical web documents, not synthetic exercises.
- [CodeParrot clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean), repository source files with license metadata.
- [Mistral v0.1 tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.1), 32k vocabulary. Download no model weights.

The preparation script pins revisions and records downloaded file hashes. Keep exact document hashes and source-group hashes. Split globally by web host or code repository before windowing, and deduplicate content globally. Qualification samples from a small number of source files are convenience samples, not a representative benchmark. TEST is prepared but never used for tuning. A full experiment needs a larger multi-shard sample, a near-duplicate audit and an immutable mixture/order manifest.

Keep raw text, source token IDs, tokenizer assets and source provenance outside Git. Publish only aggregate manifests with source revisions, counts and file digests. Preserve original per-file code licenses locally; the experiment's code license does not relicense the corpus.
