"""Arithmetic fixtures for fair hybrid serving and label-independent input splits."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from tests.test_text_classifier import TextClassifierTests, np


@unittest.skipUnless(np is not None,'NumPy required')
class NewsShiftTests(TextClassifierTests):
    # Reuse the native-library setup and fixture helper, without inheriting the
    # unrelated classifier test methods (see load_tests below).
    def hybrid_fixture(self, parent, faiss=False):
        from tools.export_news_classifier import export_seed
        from tools.external_cotraining import file_hash
        from tools.text_classifier import ngrams
        common,checkpoint = parent/'common',parent/'seed-3'
        common.mkdir(); checkpoint.mkdir()
        tokens = ['ab','ac']
        grams = sorted({g for w in tokens for g in ngrams(w,3)})
        lookup = {g:i+2 for i,g in enumerate(grams)}
        bags = [[i]+[lookup[g] for g in ngrams(w,3)] for i,w in enumerate(tokens)]
        offsets = np.r_[0,np.cumsum([len(b) for b in bags])].astype('<u8')
        offsets.tofile(common/'word-offsets.u64')
        np.concatenate(bags).astype('<u8').tofile(common/'word-ids.u64')
        rows = np.random.default_rng(42).normal(0,.1,(len(grams)+2,64)).astype('<f4')
        rows.tofile(checkpoint/'source.f32')
        np.savez(checkpoint/'head.npz',weight=np.random.default_rng(11).normal(0,.1,(4,64)).astype('<f4'),
                 bias=np.asarray([0.,.1,-.1,.2],'<f4'))
        (common/'lexicon.json').write_text(json.dumps({'tokens':tokens,'grams':grams,
            'whole_words':tokens,'whole_word_rows':2,'max_order':3,'max_tokens':128}))
        (common/'data.json').write_text(json.dumps({'rows':len(rows),'dim':64,'whole_word_rows':2}))
        training = {'status':'passed','source_sha256':file_hash(checkpoint/'source.f32'),
            'head_sha256':file_hash(checkpoint/'head.npz'),'source_commit':'arithmetic-fixture',
            'seed':3,'selected_epoch':1,'settings':{'activation':'relu'}}
        (checkpoint/'training.json').write_text(json.dumps(training))
        backends = ['source-mmap','int8','ngram-bounded','ngram-int8','precomputed-words']
        if faiss:
            backends.extend(('faiss-sq4','faiss-sq8'))
        export_seed(common,checkpoint,parent/'serving',library=self.library,backends=backends)
        return common,checkpoint

    def test_hybrids_same_known_features_and_rebased_oov(self):
        from tools.export_news_hybrids import export_hybrids
        from tools.text_classifier import TextClassifier
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            common,checkpoint=self.hybrid_fixture(parent)
            backends=['precomputed-hybrid-source','precomputed-hybrid-int8','precomputed-hybrid-ngram-int8']
            export_hybrids(common,checkpoint,parent/'serving',parent/'hybrids',backends)
            texts=['ab ac','abc','zz','ab abc','abc abc ac']
            with TextClassifier(parent/'serving/source-mmap',native_library=self.library) as source:
                expected=source.predict(texts)
            for backend in backends:
                root=parent/'hybrids'/backend
                self.assertFalse((root/'word-ids.u64').exists())
                self.assertFalse((root/'word-offsets.u64').exists())
                with TextClassifier(root,native_library=self.library) as model:
                    actual=model.predict(texts)
                    self.assertEqual(actual['oov_counts'].tolist(),[0,1,1,1,2])
                    self.assertEqual(actual['logits'][0].tobytes(),expected['logits'][0].tobytes())
                    self.assertFalse(actual['source_prediction_preserved'].any())
                    with self.assertRaisesRegex(ValueError,'precomputed'):
                        model.word_bag('ab')
                if backend=='precomputed-hybrid-source':
                    self.assertEqual(actual['logits'].tobytes(),expected['logits'].tobytes())
                else:
                    ordinary='int8' if backend.endswith('hybrid-int8') else 'ngram-int8'
                    with TextClassifier(parent/'serving'/ordinary,native_library=self.library) as model:
                        original=model.predict(['abc','zz'])
                    self.assertEqual(actual['logits'][1:3].tobytes(),original['logits'].tobytes())
            manifest=json.loads((parent/'hybrids/export.json').read_text())
            self.assertEqual(manifest['original_whole_word_rows_removed'],2)
            (parent/'serving/precomputed-words/word-vectors.f32').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                export_hybrids(common,checkpoint,parent/'serving',parent/'bad',backends)

    def test_hybrid_sq4_reuses_exact_codebook_and_unknown_codes(self):
        try:
            import faiss
        except ImportError:
            self.skipTest('FAISS required')
        faiss.omp_set_num_threads(1)
        from tools.export_news_hybrids import export_hybrids
        from tools.external_cotraining import file_hash
        from tools.text_classifier import TextClassifier
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)
            common,checkpoint=self.hybrid_fixture(parent,faiss=True)
            export_hybrids(common,checkpoint,parent/'serving',parent/'hybrids',
                           ['precomputed-hybrid-faiss-sq4','precomputed-hybrid-faiss-sq8'])
            for codec in ('faiss-sq4','faiss-sq8'):
                left=parent/'serving'/codec
                right=parent/'hybrids'/('precomputed-hybrid-'+codec)
                self.assertEqual(file_hash(left/'memory/codec.faiss'),file_hash(right/'memory/codec.faiss'))
                with TextClassifier(left) as original,TextClassifier(right) as hybrid:
                    self.assertEqual(original.predict(['abc','zz'])['logits'].tobytes(),
                                     hybrid.predict(['abc','zz'])['logits'].tobytes())

    def test_stratified_hash_split_dedup_overlap_and_literal_categories(self):
        from tools.benchmark_news_shift import prepare_requests,content_hash
        from tools.text_classifier import tokenize
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);common=parent/'common';common.mkdir()
            (common/'lexicon.json').write_text(json.dumps({'tokens':['news'],'grams':['ne'],
                'max_order':3,'max_tokens':128}))
            ag=parent/'ag.csv'
            with ag.open('w') as stream:
                csv.writer(stream).writerow(['1','Overlap','article'])
            def suffix(i):
                return chr(97+i//676)+chr(97+(i//26)%26)+chr(97+i%26)
            rows=[]
            for phase,when in enumerate(('2016-12-31','2017-01-01')):
                for i in range(1500):
                    rows.append({'category':'SPORTS','headline':'news '+suffix(i+phase*1500),
                        'short_description':'','date':when})
            rows.extend([{'category':'TECH','headline':rows[0]['headline'],'short_description':'','date':'2018-01-01'},
                {'category':'WORLD NEWS','headline':'Overlap','short_description':'article','date':'2018-01-01'},
                {'category':'POLITICS','headline':'valid news','short_description':'','date':'2018-01-01'},
                {'category':'TECH','headline':'123','short_description':'','date':'2018-01-01'}])
            raw=parent/'huffpost.jsonl'
            raw.write_text(''.join(json.dumps(r)+'\n' for r in reversed(rows)))
            result=prepare_requests(raw,ag,common,parent/'requests',verify_hashes=False)
            self.assertEqual(result['counts']['duplicate_rows'],1)
            self.assertEqual(result['counts']['conflicting_category_duplicate_rows'],1)
            self.assertEqual(result['counts']['ag_input_overlap_rows'],1)
            self.assertEqual(result['counts']['empty_rows'],1)
            self.assertEqual(result['phase_stats']['validation']['documents'],600)
            self.assertEqual(result['phase_stats']['test']['documents'],2400)
            self.assertEqual(result['phase_stats']['validation']['class_1'],600)
            self.assertEqual(result['phase_stats']['test']['class_1'],2400)
            audit=json.loads((parent/'requests/preprocessing-audit.json').read_text())
            self.assertTrue(all('text' not in r and 'headline' not in r for r in audit))
            self.assertNotIn(content_hash(tokenize('Overlap article')[0]),{r['content_sha256'] for r in audit})
            hashes=sorted(r['content_sha256'] for r in audit)
            self.assertEqual(sorted(r['content_sha256'] for r in audit if r['phase']=='validation'),hashes[:600])
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                prepare_requests(raw,ag,common,parent/'bad')

    def test_quality_gates_include_macro_f1_and_frozen_budget(self):
        from tools.benchmark_news_shift import select
        def candidate(mib,accuracy=.9,f1=.9,ce=.3,rate=100):
            return {'complete_model_bytes':mib*(1<<20),'peak_rss_bytes':200*(1<<20),
                'metrics':{'accuracy':accuracy,'macro_f1':f1,'cross_entropy':ce},
                'timings':{'32':{'documents_per_second':rate}}}
        records={'source-mmap':candidate(160),'hybrid':candidate(40,rate=500),
            'zero-oov':candidate(16,f1=.88,rate=1000),'bad-ce':candidate(16,ce=.31,rate=2000)}
        result=select(records)
        self.assertIsNone(next(v for v in result if v['budget_kind']=='artifact' and v['budget_mib']==32)['selected'])
        self.assertEqual(next(v for v in result if v['budget_kind']=='artifact' and v['budget_mib']==48)['selected'],'hybrid')

    def test_paired_bootstrap_resamples_documents_and_recomputes_draws(self):
        from tools.bootstrap_news_shift import bootstrap
        from tools.analyze_news_shift import validate_bootstrap
        source={'labels':[0,1,2,3],'predictions':[0,0,2,0],'token_counts':[10]*4,'oov_counts':[0,1,2,0]}
        runs=[{'test':{'source-mmap':source,'other':{**source,'predictions':predictions}}}
              for predictions in ([0,1,2,3],[0,1,0,3],[0,0,2,3])]
        result=bootstrap(runs,'other')
        self.assertEqual(result['documents'],4)
        self.assertEqual(result['histogram'],[0,0,1,1,0,1,1])
        self.assertAlmostEqual(result['difference_pp'],100/3)
        validate_bootstrap(runs,'other','source-mmap',result,False)
        sliced=bootstrap(runs,'other',oov=True)
        self.assertEqual(sliced['documents'],2)
        validate_bootstrap(runs,'other','source-mmap',sliced,True)
        result['ci95_pp']=[99.,100.]
        with self.assertRaisesRegex(ValueError,'confidence interval'):
            validate_bootstrap(runs,'other','source-mmap',result,False)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(NewsShiftTests(name) for name in NewsShiftTests.__dict__ if name.startswith('test_'))


if __name__=='__main__':
    unittest.main()
