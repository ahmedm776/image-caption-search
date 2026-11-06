import pickle, numpy as np
G=pickle.load(open('guard.pkl','rb'))
def mm(x): x=np.array(x); return (x-x.min())/(x.max()-x.min()+1e-12)
def order(g,m):
    n=len(g['idx']); b=np.array(g['bm25']); d=np.array(g['dense'])
    if m=='M0': sc=b
    elif m=='M1': sc=d
    elif m=='M2':
        rb=np.empty(n); rb[np.argsort(-b,kind='stable')]=np.arange(1,n+1); rd=np.empty(n); rd[np.argsort(-d,kind='stable')]=np.arange(1,n+1); sc=1/(60+rb)+1/(60+rd)
    else: sc=0.5*mm(d)+0.5*mm(b)
    return [g['idx'][i] for i in np.argsort(-sc,kind='stable')]
print('n queries',len(G))
for m in ['M0','M1','M2','M3']:
    t1=t5=0; rr=[]
    for g in G:
        o=order(g,m); r=o.index(g['target'])+1 if g['target'] in o else None
        t1+=r==1; t5+=bool(r and r<=5); rr.append(1/r if r else 0)
    print(m,'top1',t1,'top5',t5,'MRR',round(np.mean(rr),3),'target-not-in-bm25-top200',sum(g['target'] not in g['idx'] for g in G))
