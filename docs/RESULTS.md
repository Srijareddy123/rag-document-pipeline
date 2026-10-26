# Results detail

All numbers come from scripts in this repo. Raw output is in `results/`.

## Load test

50,000 articles of Simple English Wikipedia (77.7 MB of text), local `all-MiniLM-L6-v2` embeddings, fixed-size chunks, ChromaDB on disk.

```bash
make corpus     # one-time download, about 157 MB, stored outside the repo
make loadtest   # /usr/bin/time -l python scripts/load_test.py --n 50000 --strategy fixed
```

Machine: Apple M4, 16 GB, shared with other jobs; embeddings ran on the Apple GPU (MPS). Earlier runs took 585 s and 816 s, so the time is one measurement. No CPU-only run.

## Chunking comparison

3,000 articles indexed per strategy, 1,000 questions. A question is a sentence from an article ("verbatim") or the same sentence with 40 percent of its words removed ("dropout"). A hit means a top-5 chunk from the right article that contains at least 80 percent of the sentence's words.

```bash
make eval       # python scripts/compare_chunking.py --docs 3000 --queries 1000
```

| Strategy | Chunks | Verbatim hit@1 | Verbatim hit@5 | Dropout hit@1 | Dropout hit@5 |
|---|---|---|---|---|---|
| fixed (1000 chars, 200 overlap) | 16,466 | 0.521 | 0.691 | 0.340 | 0.499 |
| semantic (paragraph and sentence) | 12,823 | 0.536 | 0.679 | 0.360 | 0.519 |
| recursive (LangChain splitter) | 17,725 | 0.554 | 0.707 | 0.350 | 0.503 |

Reruns of this command moved individual numbers by about 2 points (the embedding model runs on the GPU), and with 1,000 questions differences of a few points are noise. The three strategies are within about 4 points of each other on every measure, so none clearly wins. Semantic gets similar hit rates with 22 percent fewer chunks to embed and store.

## Design choices, in full

- **Quality checks before chunking.** Embedding is the slow step, so rejecting empty, symbol-heavy and duplicate documents first saves the most time. In the 50,000 article run it dropped 501 documents before they cost any embedding time.
- **"Semantic" means structure, not embeddings.** It splits on paragraphs and sentences and merges up to 1,200 characters. I considered splitting where embedding similarity drops, and kept the cheaper version because the comparison above shows little to gain from chunking alone on this corpus.
- **Batches of documents, not one big list.** The first version chunked the whole corpus into memory before embedding. Batches keep memory flat; the 50,000 article run peaked at 749 MB resident.
- **ChromaDB over a hosted vector database.** It runs in-process or as a container, so the demo needs no account. I did not benchmark past about 125,000 vectors.
- **LangChain for one thing.** I use its recursive text splitter as a reference strategy. The rest of the pipeline is plain Python so each step can be read and tested.
