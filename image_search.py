"""Separate AI-assisted post-coursework caption search. No image pixels fetched."""
import argparse,hashlib,json,re,time,html
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,ENGLISH_STOP_WORDS
import scipy.sparse as _sp
from flask import Flask,request,render_template_string
STEM=PorterStemmer()
EXPANSIONS={'car':['vehicle','automobile'],'bike':['bicycle','motorcycle'],'dog':['puppy','animal'],'cat':['kitten','animal'],'kitchen':['stove','cabinet'],'street':['road','sidewalk'],'person':['people','man','woman'],'food':['meal','fruit'],'bathroom':['toilet','sink','shower']}
@lru_cache(maxsize=100000)
def stem(word):return STEM.stem(word)
def tokenize(text):return [stem(w) for w in re.findall('[a-z]+',str(text).lower()) if len(w)>2 and w not in ENGLISH_STOP_WORDS]
def safe_url(value):
    try:
        x=urlsplit(str(value).strip())
        return x.scheme in ('https','http') and bool(x.hostname) and x.username is None and x.password is None and not any(c.isspace() for c in str(value))
    except ValueError:return False

def load_rows(path):
    raw=pd.read_csv(path)
    if not {'caption','url'}.issubset(raw):raise ValueError('Expected caption and url columns')
    valid_caption=raw.caption.notna() & raw.caption.astype('string').str.strip().ne('')
    valid_url=raw.url.map(safe_url)
    accepted=raw.loc[valid_caption & valid_url].copy();accepted['id']=['IMG_'+str(i+1) for i in accepted.index]
    duplicates=int(accepted.duplicated(['url','caption']).sum());accepted=accepted.drop_duplicates(['url','caption'])
    if accepted.empty:raise ValueError('No accepted rows')
    report={'input_rows':len(raw),'missing_or_blank_caption':int((~valid_caption).sum()),'invalid_or_nonHTTP_url':int((~valid_url).sum()),'rejected_union':int((~(valid_caption & valid_url)).sum()),'exact_duplicate_pairs_removed':duplicates,'accepted_rows':len(accepted),'row_policy':'Sequential reject union, then exact caption+URL duplicate removal. Stable original row IDs retained; HTTP links preserved and explicitly labeled, not silently rewritten. No URL/image requests.'}
    return accepted.to_dict('records'),report

def bm25_matrix(X,k1=1.5,b=0.75):
    dl=np.asarray(X.sum(axis=1)).ravel();avgdl=dl.mean() if X.shape[0] else 1.0
    df=np.asarray((X>0).sum(axis=0)).ravel();N=X.shape[0]
    idf=np.log(1+(N-df+0.5)/(df+0.5))
    Xc=X.tocoo()
    w=(Xc.data*(k1+1))/(Xc.data+k1*(1-b+b*dl[Xc.row]/avgdl))*idf[Xc.col]
    return _sp.csr_matrix((w,(Xc.row,Xc.col)),shape=Xc.shape)

_ENC=[]
def default_encoder():
    """Lazy-load all-MiniLM-L6-v2 (needs sentence-transformers; optional dependency, downloads the model once)."""
    if not _ENC:
        from sentence_transformers import SentenceTransformer
        m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu');m.max_seq_length=128
        _ENC.append(lambda texts:m.encode(texts,batch_size=64,normalize_embeddings=True))
    return _ENC[0]

class Engine:
    """BM25 over stemmed unigram+bigram features. Scores are BM25 weights, not probabilities."""
    def __init__(self,rows,k1=1.5,b=0.75):
        self.rows=rows;self.k1=k1;self.b=b
        self.vectorizer=CountVectorizer(tokenizer=tokenize,token_pattern=None,lowercase=False,ngram_range=(1,2))
        counts=self.vectorizer.fit_transform(r['caption'] for r in rows).tocsr().astype(np.float64)
        self.matrix=bm25_matrix(counts,k1,b)
    def query(self,q,k=5,expand=False,semantic=False,encoder=None):
        q=str(q).strip()
        if len(q)>500:raise ValueError('Query limit500characters')
        if k<1 or k>50:raise ValueError('k must be1..50')
        # Same stemmer applied to both original and expansion text, no set-order randomness.
        if expand:
            words=re.findall('[a-z]+',q.lower());q+=' '+' '.join(w for t in words for w in EXPANSIONS.get(t,[]))
        v=self.vectorizer.transform([q])
        if not v.nnz:return []
        scores=(self.matrix@v.T).toarray().ravel()
        idx=np.flatnonzero(scores>0);order=sorted(idx,key=lambda i:(-scores[i],self.rows[i]['id']))[:k]
        if semantic:return self._semantic_rerank(q,scores,idx,k,encoder)
        return [{'id':self.rows[i]['id'],'score':float(scores[i]),'caption':self.rows[i]['caption'],'url':self.rows[i]['url']} for i in order]

    CANDIDATES=200
    def _semantic_rerank(self,q,scores,idx,k,encoder=None):
        """OPTIONAL. Rerank the BM25 top-200 by 0.5*minmax(dense cosine)+0.5*minmax(BM25). Cannot surface documents BM25 misses.
        Measured on the author's frozen protocol (eval/SEMANTIC_RERANK.md); slower (embeds 200 captions per query on CPU)."""
        enc=encoder or default_encoder()
        order=sorted(idx,key=lambda i:(-scores[i],self.rows[i]['id']))[:self.CANDIDATES]
        e=np.asarray(enc([self.rows[i]['caption'] for i in order]+[str(q)]),dtype=np.float64)
        dense=e[:-1]@e[-1];bm=np.array([scores[i] for i in order])
        mm=lambda x:(x-x.min())/(x.max()-x.min()+1e-12)
        fused=0.5*mm(dense)+0.5*mm(bm)
        pos=sorted(range(len(order)),key=lambda p:(-fused[p],self.rows[order[p]]['id']))[:k]
        return [{'id':self.rows[order[p]]['id'],'score':float(fused[p]),'caption':self.rows[order[p]]['caption'],'url':self.rows[order[p]]['url']} for p in pos]

TEMPLATE='''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Caption search</title><style>body{font:16px Arial;background:#f3f6f8;color:#18333e;margin:0}main{max-width:940px;margin:auto;padding:28px}h1{margin-bottom:8px}.note{padding:14px;background:#fff1cf;border-radius:10px}form,article{padding:20px;background:white;border:1px solid #d8e2e7;border-radius:12px;margin:18px 0}input[type=text]{box-sizing:border-box;width:100%;padding:12px;font-size:17px;margin:8px 0 14px}button{background:#14666c;color:white;border:0;padding:12px 20px;border-radius:7px;margin-top:12px}.small{font-size:13px;color:#5c717b}a{color:#09676d;overflow-wrap:anywhere}p{line-height:1.5}.score{font-size:13px;color:#546d76}</style></head><body><main><p class="small">DSAI201 / PRIVATE POST-COURSEWORK RECOVERY</p><h1>Search image captions</h1><p>Text BM25 ranking over caption words, not visual recognition or a relevance benchmark.</p><p class="note">{{ banner }} No images loaded automatically. Image links may be stale or unsuitable; opening one visits an outside site.</p><form method="get"><label for="q">Caption query</label><input id="q" name="q" type="text" maxlength="500" value="{{ q }}" placeholder="kitchen wooden cabinet"><label><input name="expand" type="checkbox" value="1" {% if expand %}checked{% endif %}> Optional manual synonym expansion (changes query, unbenchmarked)</label><br><label><input name="semantic" type="checkbox" value="1" {% if semantic %}checked{% endif %}> Optional semantic rerank of BM25 top 200 (slower, needs sentence-transformers, see eval/SEMANTIC_RERANK.md)</label><br><button>Search captions</button></form><p>{{ message }}</p>{% for row in results %}<article><h2>{{ row.id }}</h2><p class="score">{% if fused %}Fused BM25+semantic rank score (0-1 within the top-200 candidates){% else %}BM25 score{% endif %} {{ '%.4f'|format(row.score) }} (not probability)</p><p>{{ row.caption }}</p>{% if safe(row.url) %}<a href="{{ row.url }}" target="_blank" rel="noopener noreferrer nofollow">Open original image URL ({{ row.url.split(':')[0]|upper }})</a>{% endif %}</article>{% endfor %}<p class="small">Coursework personal contribution unconfirmed. Separate version AI-assisted. Source data rights and external image rights are separate. No public publication.</p></main></body></html>'''
def app_for(engine,demo=False):
    app=Flask(__name__)
    @app.get('/')
    def home():
        q=request.args.get('q','');expand=request.args.get('expand')=='1';semantic=request.args.get('semantic')=='1';results=[];message='Enter a query to search.';fused=False
        try:
            if q.strip():
                try:results=engine.query(q,expand=expand,semantic=semantic);fused=semantic
                except ValueError:raise
                except Exception as err:
                    if not semantic:raise
                    results=engine.query(q,expand=expand);fused=False;note=' Semantic rerank unavailable ('+type(err).__name__+'); showing plain BM25 results instead.'
                else:note=''
                message=(f'{len(results)} matching results shown.' if results else 'No matching captions. Try different words.')+note
        except ValueError as e:message=str(e)
        response=app.make_response(render_template_string(TEMPLATE,q=q,expand=expand,semantic=semantic,fused=fused,results=results,message=message,safe=safe_url,banner='SYNTHETIC TEST FIXTURE - not original dataset.' if demo else 'Current Kaggle caption corpus. Links and captions are unverified external content.'))
        response.headers['Content-Security-Policy']="default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        response.headers['X-Content-Type-Options']='nosniff';return response
    return app

def run(path,out):
    start=time.perf_counter();rows,audit=load_rows(path);engine=Engine(rows);build=time.perf_counter()-start
    checks={}
    for query in ['kitchen cabinet','bathroom sink','motorcycle','street bicycle','cat cabinet','xyzzqqqq']:
        t=time.perf_counter();r=engine.query(query);checks[query]={'ids':[x['id'] for x in r],'scores':[x['score'] for x in r],'seconds':time.perf_counter()-t}
    report={**audit,'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'bm25_shape':list(engine.matrix.shape),'ranking':'BM25(k1=1.5,b=0.75) over stemmed unigram+bigram features','nnz':int(engine.matrix.nnz),'build_seconds':build,'queries':checks,'evaluation':'Development sanity/timing only. No relevance judgments, held-out metric, accuracy or speed superiority claim. No external image URL checked.'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'run_report.json').write_text(json.dumps(report,indent=2));return engine,report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--output',default='local-output');p.add_argument('--serve',action='store_true');p.add_argument('--demo',action='store_true');p.add_argument('--port',type=int,default=8052);a=p.parse_args();engine,report=run(a.csv,a.output);print(json.dumps(report,indent=2),flush=True)
    if a.serve:app_for(engine,a.demo).run(host='127.0.0.1',port=a.port,debug=False)
