import os, sys, random, pickle, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from image_search import load_rows, Engine
from sentence_transformers import SentenceTransformer
P = os.environ.get('CAPTIONS_CSV', 'train.csv')  # path to the Kaggle 220k captions CSV (see download_data.py)
rows, _ = load_rows(P); eng = Engine(rows)
ex = {r['id'] for r in random.Random(42).sample(rows, 150)} | {r['id'] for r in random.Random(2026).sample(rows, 150)}
pool = [r for r in rows if r['id'] not in ex]
samp = random.Random(7777).sample(pool, 150)
assert not ({r['id'] for r in samp} & ex)
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); m.max_seq_length = 128
idmap = {r['id']: i for i, r in enumerate(rows)}; out = []
for n, r in enumerate(samp):
    q = ' '.join([w for w in r['caption'].split() if len(w) > 3][:8])
    v = eng.vectorizer.transform([q]); s = (eng.matrix @ v.T).toarray().ravel()
    idx = [int(i) for i in np.argsort(-s, kind='stable')[:200] if s[i] > 0]
    e = m.encode([rows[i]['caption'] for i in idx] + [q], batch_size=64, normalize_embeddings=True)
    out.append(dict(target=idmap[r['id']], idx=idx, bm25=[float(s[i]) for i in idx], dense=(e[:-1] @ e[-1]).tolist()))
    if n % 10 == 0: print(n, flush=True)
    pickle.dump(out, open('guard.pkl', 'wb'))
print('DONE', flush=True)
