# Pre-registration (written 2026-11-05 ~16:15 EET, after candidate retrieval for the queries was computed but BEFORE any ranking comparison, pooling or judging)
Scope: reranking WITHIN the top-200 BM25 candidates (baseline = unmodified current BM25+bigram engine, package sha256 6e3c83a2b13f5b25...).
It cannot recover documents BM25 misses, so it is NOT a test of full-corpus dense retrieval.
Methods: M0 BM25 order. M1 dense cosine (all-MiniLM-L6-v2, 128 tokens) order. M2 reciprocal-rank fusion of BM25 and dense ranks (k=60, no parameter).
M3(w) = w*minmax(dense) + (1-w)*minmax(bm25) over the 200 candidates, w in {0.25,0.5,0.75}.
Queries: 10 dev + 20 holdout (frozen as eval/semantic/topical_queries.json in this repo before any retrieval). Dev picks w only. Holdout judged once.
Relevance rubric (fixed now): a caption is relevant if its text states the query's subject AND the action/setting/attribute the query asks for; unstated attributes (e.g. gender, weather not mentioned) do not count; subject present but action/setting missing = not relevant.
Judging: pooled union of top-5 of each method, shuffled, method identity hidden, labels 1/0 written from caption text only. Author-built, single judge (an AI agent), not independent.
Metrics: P@5 and P@1 per query and mean. Per-query wins/losses vs M0 reported.
Known-item regression guard: new 150-caption sample, seed 7777, disjoint from the seed-42 and seed-2026 samples; query = first 8 content words (>3 chars); metrics top-1, top-5, MRR for M0 vs M1, M2, M3(w_dev).
MATERIAL REGRESSION (defined now): known-item top-1 falls by >= 5 captions of 150 OR MRR falls by >= 0.03 vs M0; or holdout topical mean P@5 falls by >= 0.05 vs M0.
BENEFIT (defined now): holdout mean P@5 higher than M0 by >= 0.05 AND no material regression. Anything else is reported as "no demonstrated benefit", with all tradeoffs kept.
Non-claims: no independent validation, no generalization beyond these queries, no full-corpus recall claim, no rating change.
