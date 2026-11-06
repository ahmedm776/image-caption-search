import os
import pickle, json, random, numpy as np
D = pickle.load(open('cands.pkl', 'rb')); O = D['out']; caps = D['caps']; ids = D['ids']
def mm(x): x = np.array(x); return (x - x.min()) / (x.max() - x.min() + 1e-12)
def ranks(q, method):
    o = O[q]; n = len(o['idx']); b = np.array(o['bm25']); d = np.array(o['dense'])
    if method == 'M0': sc = b
    elif method == 'M1': sc = d
    elif method == 'M2':
        rb = np.empty(n); rb[np.argsort(-b, kind='stable')] = np.arange(1, n + 1)
        rd = np.empty(n); rd[np.argsort(-d, kind='stable')] = np.arange(1, n + 1)
        sc = 1 / (60 + rb) + 1 / (60 + rd)
    else:
        w = float(method[3:]) ; sc = w * mm(d) + (1 - w) * mm(b)   # M3_0.25 etc
    order = np.argsort(-sc, kind='stable')
    return [o['idx'][i] for i in order]
def pool(queries, methods, seed, depth=5):
    rng = random.Random(seed); blind = {}; key = {}
    for q in queries:
        u = sorted({i for m in methods for i in ranks(q, m)[:depth]})
        rng.shuffle(u); blind[q] = [{'item': k + 1, 'caption': caps[i]} for k, i in enumerate(u)]
        key[q] = {k + 1: i for k, i in enumerate(u)}
    return blind, key
if __name__ == '__main__':
    import sys
    Q = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topical_queries.json')))
    blind, key = pool(Q['dev'], ['M0', 'M1', 'M2', 'M3_0.25', 'M3_0.5', 'M3_0.75'], 11)
    json.dump(blind, open('dev_pool_blind.json', 'w'), indent=1); json.dump(key, open('dev_pool_key.json', 'w'))
    print({q: len(v) for q, v in blind.items()})
