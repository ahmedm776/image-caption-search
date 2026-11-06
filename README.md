# Image Caption Search with Semantic Re-ranking

Search over image-caption TEXT (not image pixels): a BM25 sparse baseline over
217,868 captions, plus an optional semantic re-ranking stage evaluated under a
pre-registered held-out protocol. Pairs a preserved original DSAI201 coursework
notebook (source-only) with a later, substantially extended search module and
Gradio UI.

## Contents

- `original_source_only.ipynb` - original coursework notebook, source-only
  (historical outputs stripped).
- `image_search.py` - improved search module and Gradio UI. UI screenshots
  (`desktop.png`, `mobile.png`, `empty.png`) use invented synthetic captions
  only.
- `download_data.py` - fetches the caption dataset from its canonical source.
- `eval/` - evaluation records: pre-registration (`semantic/PREREGISTRATION.md`),
  dev and held-out protocols, judgment sets and pools for independent
  re-checking. The judgment files contain a small subset of public-dataset
  caption strings used as judgment keys; the full corpus, generated index and
  model state are not redistributed.
- `run_report.json`, `verification.json`, `baseline_status.json`,
  `tests.log`, `ui_tests.log`, `real_ui_tests.log` - executed-run records.
- `test_search.py` - unit tests on synthetic fixtures.

## Data and rights

- Dataset: `thedevastator/220k-gpt4vision-image-captions` on Kaggle (version 2;
  upstream: LAION `220k-GPT4Vision-captions-from-LIVIS` on Hugging Face).
  Kaggle reports CC0; upstream metadata is Apache-2.0. The upstream card
  describes the captions as Mistral-7B-OpenOrca summaries of LVIS-Instruct4V
  dialogues. External COCO image rights remain separate; no images are
  fetched or redistributed here.

## Execution and honest limits

- The original notebook's core cells were re-executed on the current v2 data
  with only the local data path changed: 217,868 rows, 21,277 terms,
  8,078,942 tokens in 88.8 s. This is not a full original run and not proof of
  historical-file identity; the Gradio launch, download cell and pickle
  save/load cells were omitted.
- Whole-corpus runs of the improved source repeat with identical acceptance
  checks, matrices, result IDs and scores (timings excepted); see
  `verification.json`.
- Evaluation numbers in `eval/` are development-exposed benchmark reuse unless
  a held-out protocol is named; the semantic stage was pre-registered before
  held-out retrieval (see `eval/semantic/PREREGISTRATION.md`).

## Attribution

- The original coursework notebook is preserved as recovered; its team/solo
  context is not recorded, so no sole-authorship claim is made for it.
- The search module, UI, tests and evaluation harness are later portfolio
  work, built with AI-assistant help and presented as such.
