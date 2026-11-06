import tempfile,unittest,json
from pathlib import Path
from image_search import *
class Tests(unittest.TestCase):
 def fixture(self):return [{'id':'1','caption':'A wooden kitchen cabinet and stove','url':'https://example.com/a'},{'id':'2','caption':'A motorcycle travels along a road','url':'https://example.com/b'}]
 def test_rank(self):self.assertEqual(Engine(self.fixture()).query('cabinet')[0]['id'],'1')
 def test_empty(self):self.assertEqual(Engine(self.fixture()).query(''),[])
 def test_oov(self):self.assertEqual(Engine(self.fixture()).query('xyzzzqq'),[])
 def test_expansion(self):self.assertEqual(Engine(self.fixture()).query('bike',expand=True)[0]['id'],'2')
 def test_consistent_stem(self):self.assertEqual(tokenize('motorcycles'),tokenize('motorcycle'))
 def test_bad_url(self):
  for s in ['javascript:alert(1)','https://user:pass@example.com','https://example.com/ x']:self.assertFalse(safe_url(s))
 def test_html_escape(self):
  e=Engine([{'id':'x','caption':'test <script>alert(1)</script> cabinet','url':'https://example.com/"onclick="x'}]);c=app_for(e,True).test_client();r=c.get('/?q=cabinet');self.assertNotIn(b'<script>',r.data);self.assertIn(b'&lt;script&gt;',r.data);self.assertIn('default-src',r.headers['Content-Security-Policy'])
 def test_missing_caption_duplicate(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.csv';p.write_text('caption,url\n,https://example.com/a\ncat,https://example.com/b\ncat,https://example.com/b\ncat,javascript:bad\n');rows,r=load_rows(p);self.assertEqual(len(rows),1);self.assertEqual(r['rejected_union'],2);self.assertEqual(r['exact_duplicate_pairs_removed'],1)
 def test_missing_column(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.csv';p.write_text('other\n1\n')
   with self.assertRaises(ValueError):load_rows(p)
 def test_limit(self):
  with self.assertRaises(ValueError):Engine(self.fixture()).query('x'*501)
 def test_ties(self):
  e=Engine([{'id':'z','caption':'apple fruit','url':'https://example.com/z'},{'id':'a','caption':'apple fruit','url':'https://example.com/a'}]);self.assertEqual([r['id'] for r in e.query('apple')],['a','z'])
 def test_semantic_rerank_uses_fusion_and_stays_inside_bm25_candidates(self):
  rows=[{'id':str(i),'caption':c,'url':'https://example.com/%d'%i} for i,c in enumerate(['red bus on road','bus parked near road','road with trees','green field'])]
  e=Engine(rows)
  def fake(texts):  # unit vectors: prefers caption 1 over 0 and the query direction is axis 0
   import numpy as np
   v=np.array([[0.2,0.98] if 'parked' in t else [1.0,0.0] for t in texts[:-1]]+[[0.0,1.0]])
   return v/np.linalg.norm(v,axis=1,keepdims=True)
  base=[r['id'] for r in e.query('bus road',k=3)]
  sem=[r['id'] for r in e.query('bus road',k=3,semantic=True,encoder=fake)]
  self.assertEqual(set(sem)<=set(r['id'] for r in e.query('bus road',k=50)),True)
  self.assertEqual(sem[0],'1');self.assertNotEqual(base,sem)
  self.assertEqual([r['id'] for r in e.query('xyzzzqq',semantic=True,encoder=fake)],[])
 def test_semantic_default_off(self):
  e=Engine(self.fixture());self.assertEqual(e.query('cabinet'),e.query('cabinet',semantic=False))
 def test_semantic_unavailable_falls_back_with_message_and_keeps_checkbox(self):
  import image_search
  e=Engine(self.fixture());old=image_search.default_encoder
  def boom():raise ImportError('no model')
  image_search.default_encoder=boom
  try:r=app_for(e,True).test_client().get('/?q=cabinet&semantic=1')
  finally:image_search.default_encoder=old
  self.assertEqual(r.status_code,200);self.assertIn(b'Semantic rerank unavailable (ImportError)',r.data);self.assertIn(b'BM25 score',r.data);self.assertNotIn(b'Fused',r.data);self.assertIn(b'name="semantic" type="checkbox" value="1" checked',r.data)
 def test_default_page_has_unchecked_semantic_and_bm25_label(self):
  r=app_for(Engine(self.fixture()),True).test_client().get('/?q=cabinet');self.assertNotIn(b'value="1" checked',r.data);self.assertIn(b'BM25 score',r.data)
if __name__=='__main__':unittest.main()
