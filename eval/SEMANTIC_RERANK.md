# Optional semantic rerank of the BM25 top-200 (2026-11-05)

Added as an OPTIONAL mode (off by default, UI checkbox / `semantic=True`). Default ranking is still the unmodified BM25+bigram.
Scope: it only reorders the BM25 top-200; it cannot surface a caption BM25 misses. It does NOT test full-corpus dense retrieval.
Method: 0.5*minmax(MiniLM cosine) + 0.5*minmax(BM25) over the 200 candidates (all-MiniLM-L6-v2, 128 tokens). Cost: 3-7 s per query on a 2-core CPU
(embeds 200 captions), plus an optional dependency (sentence-transformers) and a one-time model download.
Baseline: the previous package (BM25+bigram), sha256 6e3c83a2b13f5b25... Protocol and thresholds: eval/semantic/PREREGISTRATION.md (written before comparisons or judging; candidate lists were computed before it).

## Results (author-built judgments, single AI judge, not independent)
Topical queries, P@5 (relevance rubric: subject AND action/setting both stated in the caption text; unstated attributes do not count; 'clean' was not required).
| Set | BM25 (baseline) | Dense only | RRF | Fusion w=0.5 (shipped option) |
|---|---|---|---|---|
| Dev (10 queries, used to pick w=0.5) | 0.640 | 0.760 | 0.700 | 0.720 (w=0.75 tied, lower P@1) |
| Holdout (20 queries, scored once) | 0.560 | 0.610 | 0.680 | 0.650 |
Holdout fusion vs BM25 per query: 8 better, 2 worse, 10 equal; mean +0.09 (sample sd 0.22 across queries). The 2 worse queries: a tall clock tower against a blue sky (P@5 1.0 -> 0.6) and a pizza with melted cheese and pepperoni (1.0 -> 0.8). Full per-query table: `python eval/semantic/evalhold.py` (needs the candidate file from cands.py).
Known-item regression guard (new sample, seed 7777, 150 captions disjoint from the seed-42 and seed-2026 samples; first-8-content-word query):
| | top-1 | top-5 | MRR |
|---|---|---|---|
| BM25 (baseline) | 139 | 148 | 0.954 |
| Fusion w=0.5 | 139 | 147 | 0.951 |
| RRF | 113 | 139 | 0.827 |
| Dense only | 68 | 102 | 0.558 |
Pre-defined material regression (top-1 down >=5 of 150, MRR down >=0.03, or holdout P@5 down >=0.05): fusion w=0.5 has none; RRF and dense-only materially regress known-item lookup, so they are NOT shipped.
Pre-defined benefit (holdout P@5 up >=0.05 and no material regression): met by fusion w=0.5 (+0.09).

## Audit: full-text re-judgment of the holdout (added after review)
The first holdout judgments above (j_hold_*.json) were made from only the first ~200 characters of each caption, which can miss omissions later in the text. The same 217 pooled, method-blind items were re-judged on FULL caption text with the unchanged rubric (jf_*.json). 7 of 217 labels changed (all 0 -> 1: truncation under-judged). Because the holdout had by then been scored, this is an AUDIT, not a fresh independent test; the holdout is now exposed.
Full-text holdout P@5: BM25 0.59, dense-only 0.65, RRF 0.69, fusion w=0.5 0.68. Fusion vs BM25 per query: 8 better, 2 worse, 10 equal, mean +0.09 (same as truncated). The conclusion did not change, but the truncated numbers in the table above should be read as the first-pass numbers.

## Why fusion w=0.5 and not RRF
Before any holdout work, dev P@5 was RRF 0.700 vs fusion w=0.5 0.720 (w=0.75 tied at 0.720 with lower P@1, so w=0.5 was kept). Separately, the known-item guard acceptance rule was fixed in PREREGISTRATION.md before it was run: a method with top-1 down >= 5 of 150 or MRR down >= 0.03 is not accepted. RRF (top-1 -26, MRR -0.127) and dense-only fail that rule, fusion passes (top-1 0, MRR -0.003). RRF scoring higher on holdout topical P@5 did not enter the decision. All variants and tradeoffs are reported either way.
Definitions: 'material regression' = known-item top-1 down by >= 5 of 150 (3.3 points) OR known-item MRR down by >= 0.03 OR holdout topical mean P@5 down by >= 0.05, vs BM25. 'benefit' = holdout mean P@5 up by >= 0.05 AND no material regression. 'sd' everywhere = sample standard deviation (n-1) of the per-query P@5 differences versus BM25 across the 20 queries; no confidence interval is computed.

## Display and failure behavior (real corpus, checked in a browser)
With the checkbox on, results are labelled "Fused BM25+semantic rank score (0-1 within the top-200 candidates)" followed by "(not probability)"; with it off the label is "BM25 score". The checkbox keeps its state after search. If the semantic model or sentence-transformers is unavailable, the page still returns 200, shows "Semantic rerank unavailable (<ErrorName>); showing plain BM25 results instead.", labels the results BM25 and keeps the checkbox checked (unit-tested with a failing encoder; not exercised with a real missing install). The first semantic query also loads the model (about 7 s); later queries took 3-4 s.

## Limits
- 20 holdout queries, 100 judged top-5 slots; single AI judge; not independent.
- Queries are simple COCO/LVIS-style scenes written by the same author. Not evidence for other query types or other corpora.
- Known-item numbers are for first-8-word partial recall only; reranking cannot help when BM25 already finds the target and can lose it (top-5 148 -> 147).
- The dev queries were used to select w; the 'benefit' statement rests on the 20 holdout queries.
- No images are evaluated; ranking is caption TEXT only. Source data rights are unchanged: no captions, embeddings or indexes are included here (judgment files use item numbers and row indices only).
## Reproduce
Needs the source CSV (see download_data.py), sentence-transformers and nltk/flask: `python eval/semantic/cands.py` (edit paths) then `rank.py`, `evalhold.py`; `guard.py` then `evalguard.py`. Paths in these scripts are the author's sandbox paths and need editing.
