import os
import json, numpy as np
from rank import ranks
J={}
for f in 'abc': J.update(json.load(open(f'j_hold_{f}.json')))
key=json.load(open('hold_pool_key.json')); Q=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'topical_queries.json')))['holdout']
assert set(Q)==set(J) and all(len(J[q])==len(key[q]) for q in Q)
inv={q:{v:int(k) for k,v in key[q].items()} for q in Q}
M=['M0','M1','M2','M3_0.5']; P5={m:[] for m in M}; P1={m:[] for m in M}
for q in Q:
    for m in M:
        lab=[J[q][str(inv[q][i])] for i in ranks(q,m)[:5]]; P5[m].append(np.mean(lab)); P1[m].append(lab[0])
for m in M: print(m,'P@5',round(np.mean(P5[m]),3),'P@1',round(np.mean(P1[m]),2))
for m in M[1:]:
    d=np.array(P5[m])-np.array(P5['M0']); print(m,'vs M0 per-query: wins',int((d>0).sum()),'losses',int((d<0).sum()),'ties',int((d==0).sum()),'mean diff',round(d.mean(),3),'sd',round(d.std(ddof=1),3))
print('\nper-query P@5 (M0,M1,M2,M3_0.5):')
for i,q in enumerate(Q): print(q[:45].ljust(46),[round(P5[m][i],1) for m in M])
