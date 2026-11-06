import os, sys, json, pickle, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from image_search import load_rows, Engine
from sentence_transformers import SentenceTransformer
P = os.environ.get('CAPTIONS_CSV', 'train.csv')  # path to the Kaggle 220k captions CSV (see download_data.py)
rows, _ = load_rows(P); eng = Engine(rows); print('engine', eng.matrix.shape, flush=True)
Q = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topical_queries.json'))); qs = Q['dev'] + Q['holdout']
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu'); m.max_seq_length = 128
out = {}
for q in qs:
    v = eng.vectorizer.transform([q]); s = (eng.matrix @ v.T).toarray().ravel()
    idx = np.argsort(-s, kind='stable')[:200]; idx = [int(i) for i in idx if s[i] > 0]
    e = m.encode([rows[i]['caption'] for i in idx] + [q], batch_size=64, normalize_embeddings=True)
    out[q] = dict(idx=idx, bm25=[float(s[i]) for i in idx], dense=(e[:-1] @ e[-1]).tolist())
    print(q, len(idx), flush=True)
pickle.dump(dict(out=out, ids=[r['id'] for r in rows], caps=[r['caption'] for r in rows]), open('cands.pkl', 'wb'))
print('DONE', flush=True)
