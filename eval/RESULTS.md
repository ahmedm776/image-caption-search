# Retrieval evaluation

Deterministic (fixed seeds), reproducible:
`python3 eval/evaluate.py --csv <train.csv path>`.
Author-built judgments ship as data for re-checking (eval/judgments.json,
eval/judged_pool.json), they are not an independent benchmark.

## Ranker change (2026-11-04): TF-IDF cosine -> BM25 + bigram

The ranker changed from TF-IDF cosine over stemmed unigrams to BM25
(k1=1.5, b=0.75, Robertson document-frequency smoothing) over stemmed
unigram + bigram features. The same frozen evaluation protocol was rerun
before and after on the same corpus (source CSV SHA256
d23256511d8ca679e8dd34b2df94e8186a8acb93b0bb6163055a5e6adf5d617e).
Candidate variants (BM25 unigram, BM25 with optional query expansion,
TF-IDF bigram, BM25 bigram) were compared on the same protocol first;
BM25+bigram won and was integrated. Because the 150-sample and the 10
judged queries were used to SELECT the variant, they are a development set,
not held-out confirmation; see the held-out section below for the untouched
check.

Measured before/after:

| Stage | Metric | Before (TF-IDF cosine) | After (BM25+bigram) |
|---|---|---|---|
| Known-item, first-8-content-words (n=150) | top-1 | 67/150 | 140/150 |
| | top-5 | 101/150 | 150/150 |
| | MRR | 0.547 | 0.963 |
| Known-item, full caption text (n=150) | top-1 / MRR | 150/150 / 1.0 | 150/150 / 1.0 |
| Judged topical queries (10 queries, top-6 judged) | P@1 | 1.0 | 0.9 |
| | mean P@6 | 0.767 | 0.800 |

Interpretation: partial known-item recall (the realistic "I remember some
words" case) improved substantially. Judged topical quality is roughly
comparable: mean P@6 up slightly, P@1 down one query. The P@1 miss is
"a busy kitchen with people cooking" - the new top hits describe kitchens
with food but no people cooking; the one relevant hit (people eating at a
table in a restaurant kitchen) ranks 6th. Judgments for the new pool were
re-made by the same author under the same rubric after the ranker change;
the change section and both judgment sets are preserved in git/file history
of the package, not hidden.

## Held-out confirmation (untouched by variant selection)

After BM25+bigram was already adopted, a fresh disjoint sample of 150
captions (seed 2026, none overlapping the development 150) was evaluated
once, for both rankers, with the same realistic partial-query protocol
(first 8 words longer than 3 chars):

| Ranker | top-1 | top-5 | MRR |
|---|---|---|---|
| TF-IDF cosine (old) | 69/150 | 105/150 | 0.553 |
| BM25+bigram (adopted) | 137/150 | 146/150 | 0.939 |

The known-item improvement replicates on data never used for selection.
No held-out check exists for the judged topical stage (new judgments would
be needed); treat judged numbers as development-only.

## Judgment provenance (pool and judge changed)

The original TF-IDF pool and its judgments by the original judge are
preserved unchanged at eval/judged_pool_tfidf.json and
eval/judgments_tfidf.json. After the ranker change, the top-6 pools are
different captions, so the original labels cannot transfer; the NEW pools
were judged by a DIFFERENT author (the agent doing this improvement pass)
under the same written rubric, calibrated against the original judgments'
edge cases. This is a same-rubric re-judgment, not a fixed-label
comparison: the P@1 1.0 -> 0.9 and P@6 0.767 -> 0.800 numbers compare
two judges on two different pools and carry that caveat.

## Stage 1 - known-item retrieval (150 sampled captions)
(a) Full caption text as query (looking up a remembered caption):
top-1 = 150/150, MRR = 1.0.
(b) First 8 words longer than 3 chars (partial recall - the realistic case):
top-1 = 140/150, top-5 = 150/150, MRR = 0.963.
Partial queries compete against many near-identical scene descriptions in a
217k-caption corpus; (b) is intentionally hard and shows the real limit.

## Stage 2 - judged topical queries (10 natural queries)
Top-6 hits per query judged relevant/not from caption text. P@1 = 0.9,
mean P@6 = 0.800.
Observed failure modes: object-only scenes when the query asks for a subject
plus action ("a busy kitchen with people cooking" returns kitchens with no
people), and a STOP sign with a bike sticker for "a man riding a bike".
Gender-attribute queries now mostly resolve because bigrams keep the
gendered subject phrase together.

## Limits
- Judgments are AUTHOR-BUILT: one judge, the same person who built the
  system. Behavioral evidence, not an independent benchmark. Judgments and
  the captions they were made from ship as data (eval/judgments.json,
  eval/judged_pool.json) for re-checking.
- The search ranks caption TEXT only, never image pixels - relevance here
  means caption-text relevance, not visual correctness of the underlying
  images (which are never fetched).
- Caption text quality of the upstream GPT4Vision dataset is not evaluated.
- Optional query expansion is off by default and remains unbenchmarked.
