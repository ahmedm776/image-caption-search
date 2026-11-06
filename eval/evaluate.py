"""Retrieval evaluation for the caption search engine. Reproduces eval/RESULTS.md.

1. Known-item retrieval: 150 deterministically sampled captions.
   (a) full caption text as the query (remembered-caption lookup);
   (b) first 8 words longer than 3 chars (partial recall). Success = the
   source row in top-k. Captions are dense descriptions, so near-identical
   scenes compete; (b) is intentionally hard.
2. Judged topical queries: 10 natural queries, top-6 hits, relevance judged
   from caption text. Judgments are AUTHOR-BUILT (same person as the system),
   shipped as eval/judgments.json, re-checkable against eval/judged_pool.json.
   Behavioral evidence, not an independent benchmark.

Caption text quality itself (upstream GPT4Vision output) is NOT evaluated, and
search ranks caption text only - never image pixels.

Usage: python3 eval/evaluate.py --csv /path/to/train.csv
Dataset: kagglehub handle thedevastator/220k-gpt4vision-image-captions (v2).
"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from image_search import load_rows, Engine

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    args = ap.parse_args()
    rows, _ = load_rows(args.csv)
    engine = Engine(rows)
    here = Path(__file__).resolve().parent
    rng = random.Random(42)
    sample = rng.sample(rows, 150)

    def variant(make_q):
        top1 = top5 = 0; rr = []
        for row in sample:
            q = make_q(row)
            hits = engine.query(q, k=10)
            ids = [h['id'] for h in hits]
            rank = ids.index(row['id']) + 1 if row['id'] in ids else None
            top1 += rank == 1; top5 += bool(rank and rank <= 5)
            rr.append(1.0 / rank if rank else 0.0)
        return {'top1': top1, 'top5': top5, 'MRR': round(sum(rr) / len(rr), 4)}

    known = {'n': 150,
             'full_caption_query': variant(lambda r: r['caption'][:500]),
             'first8_content_words': variant(lambda r: ' '.join([w for w in r['caption'].split() if len(w) > 3][:8]))}

    pool = json.load(open(here / 'judged_pool.json'))
    judgments = json.load(open(here / 'judgments.json'))
    judgments = {k: v for k, v in judgments.items() if not k.startswith('_')}
    mismatches = []; pall = []; p1 = 0
    for q, judged in judgments.items():
        live_ids = [h['id'] for h in engine.query(q, k=6)]
        if live_ids != [h['id'] for h in pool[q]]:
            mismatches.append(q)
        vals = [judged.get(i, 0) for i in live_ids]
        pall.append(sum(vals) / len(vals))
        p1 += vals[0] if vals else 0
    out = {'known_item': known,
           'judged': {'queries': len(judgments), 'pool_mismatch_queries': mismatches,
                      'P@1': round(p1 / len(judgments), 3),
                      'P@6_mean': round(sum(pall) / len(pall), 3)}}
    print(json.dumps(out, indent=2))
    return out

if __name__ == '__main__':
    main()
