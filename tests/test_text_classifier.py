"""Arithmetic certificate fixtures and complete NumPy/C text-serving checks."""
from fractions import Fraction
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

if importlib.util.find_spec('numpy'):
    import numpy as np
    from tools.text_classifier import TextClassifier, linear_preservation, ngrams, tokenize
    from tools.cotraining_groups import BagPartition, export_certified_bags
    from tools.budgeted_bags import build_error_bounds
    from tools.external_cotraining import file_hash
else:
    np = None


@unittest.skipUnless(np is not None, 'NumPy required')
class TextClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.library = Path(cls.temp.name) / 'librows.so'
        subprocess.run(['cc','-std=c11','-O2','-fPIC','-shared',
                        str(Path(__file__).resolve().parents[1]/'native/meft_rows.c'),'-lz','-o',str(cls.library)],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fraction_oracle_linear_certificate(self):
        # Deliberately chosen arithmetic values, not trained quality evidence.
        head = np.asarray([[1.25, -.5], [-.75, .5], [0., 0.]], '<f4')
        bias = np.asarray([.125, -.125, -.25], '<f4')
        for vector in itertools.product((-2., -.25, 0., .25, 2.), repeat=2):
            y = np.asarray([vector], '<f8')
            for bound in (0., .125, 1., float('inf')):
                _, winner, flag = linear_preservation(y, np.asarray([bound]), head, bias)
                if flag[0]:
                    for delta in itertools.product((-1., 0., 1.), repeat=2):
                        if sum(d*d for d in delta) > 1 or not np.isfinite(bound):
                            continue
                        source = [Fraction(v)+Fraction(bound)*Fraction(d) for v,d in zip(vector,delta)]
                        scores = [sum(Fraction(float(w))*v for w,v in zip(row,source))+Fraction(float(b))
                                  for row,b in zip(head,bias)]
                        self.assertEqual(int(winner[0]), max(range(3),key=lambda i:scores[i]))
        _, _, flags = linear_preservation([[0.,0.]], [0.], np.zeros((2,2),'<f4'), np.zeros(2,'<f4'))
        self.assertFalse(flags[0], 'ties require refusal')
        with self.assertRaisesRegex(ValueError,'FP32'):
            linear_preservation([[1.,1.]], [0.], [[1.1,1.]], [0.])
        with self.assertRaises(ValueError):
            linear_preservation([[float('nan'),1.]], [0.], head, bias)

    def build(self, root, tie=False):
        root.mkdir()
        words = ['ab','ac']
        grams = sorted({g for w in words for g in ngrams(w,3)})
        lookup = {g:i for i,g in enumerate(grams)}
        bags = [[lookup[g] for g in ngrams(w,3)] for w in words]
        # Nonuniform arithmetic rows create real residuals in the fixture.
        rows = np.asarray([[1.+i/32., .5-i/64.] for i in range(len(grams))],'<f4')
        source = root.parent / 'fixture-source.f32'
        rows.tofile(source)
        partition = BagPartition.fit(bags,len(rows))
        export_certified_bags(rows,partition,root/'memory',native_library=self.library)
        build_error_bounds(source,root/'memory',root/'bounds',native_library=self.library)
        np.concatenate(bags).astype('<u8').tofile(root/'word-ids.u64')
        np.asarray([0,len(bags[0]),sum(map(len,bags))],'<u8').tofile(root/'word-offsets.u64')
        (root/'lexicon.json').write_text(json.dumps({'tokens':words,'grams':grams,'whole_word_rows':0,
                                                   'max_order':3,'max_tokens':128}))
        np.savez(root/'head.npz',weight=np.zeros((2,2),'<f4') if tie else np.asarray([[1.,0.],[-1.,0.]],'<f4'),
                 bias=np.zeros(2,'<f4') if tie else np.asarray([.5,-.5],'<f4'))
        metadata = {'schema_version':1,'kind':'ngram-linear-text','backend':'ngram-bounded','classes':['A','B'],
                    'rows':len(rows),'dim':2,'files_sha256':{str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file()}}
        (root/'classifier.json').write_text(json.dumps(metadata))
        return rows

    def test_full_text_prediction_flags_refusal_and_hash_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'app'
            self.build(root)
            with TextClassifier(root,native_library=self.library) as model:
                result = model.predict(['AB ac','abc','zz'])
                self.assertEqual(result['predictions'].tolist(),[0,0,0])
                self.assertEqual(result['token_counts'].tolist(),[2,1,1])
                self.assertEqual(result['oov_counts'].tolist(),[0,1,1])
                self.assertTrue(result['source_prediction_preserved'].all())
                model.predict(['ab'],require_preserved=True)
                with self.assertRaises(ValueError):
                    model.head[:] = 0
                with self.assertRaises(ValueError):
                    model.predict(['1234'])
                self.assertTrue(model.predict(['ab '*129])['truncated'][0])
            tie = Path(tmp)/'tie'
            self.build(tie,tie=True)
            with TextClassifier(tie,native_library=self.library) as model:
                with self.assertRaisesRegex(ValueError,'certificate'):
                    model.predict(['ab'],require_preserved=True)
            (root/'lexicon.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash'):
                TextClassifier(root,native_library=self.library)

    def test_cli_and_import_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'app'
            self.build(root)
            result = subprocess.run([sys.executable,'-m','ngram_store','classify','--store',str(root),
                '--native-library',str(self.library),'--text','ab','abc'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len(json.loads(result.stdout)['predictions']),2)
            result = subprocess.run([sys.executable,'-c',
                'import sys; from ngram_store import TextClassifier; assert "torch" not in sys.modules'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_tokenization_boundary(self):
        self.assertEqual(tokenize("Don't STOP, 123."),(["don't","stop"],False))
        with self.assertRaises(ValueError):
            tokenize(None)

    def test_complete_export_controls_share_source_and_head(self):
        from tools.export_news_classifier import export_seed
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            fixture = parent/'fixture'
            rows = self.build(fixture)
            common, checkpoint = parent/'common', parent/'seed-3'
            common.mkdir(); checkpoint.mkdir()
            import shutil
            for name in ('lexicon.json','word-ids.u64','word-offsets.u64'):
                shutil.copyfile(fixture/name,common/name)
            (common/'data.json').write_text(json.dumps({'rows':len(rows),'dim':2,'whole_word_rows':0}))
            rows.tofile(checkpoint/'source.f32')
            shutil.copyfile(fixture/'head.npz',checkpoint/'head.npz')
            training = {'status':'passed','source_sha256':file_hash(checkpoint/'source.f32'),
                        'head_sha256':file_hash(checkpoint/'head.npz'),'source_commit':'arithmetic-fixture',
                        'seed':3,'selected_epoch':1,'classes':['A','B']}
            (checkpoint/'training.json').write_text(json.dumps(training))
            backends = ['source-resident','source-mmap','linear-fused','fp16','int8','ngram-bounded','ngram-int8','precomputed-words']
            export_seed(common,checkpoint,parent/'serving',library=self.library,backends=backends)
            results = {}
            for backend in backends:
                with TextClassifier(parent/'serving'/backend,native_library=self.library) as model:
                    results[backend] = model.predict(['ab','ac','abc'])
                self.assertEqual(results[backend]['predictions'].tolist(),[0,0,0])
            self.assertEqual(results['source-resident']['logits'].tobytes(),results['source-mmap']['logits'].tobytes())
            self.assertEqual(results['source-mmap']['logits'][:2].tobytes(),results['precomputed-words']['logits'][:2].tobytes())
            self.assertFalse(results['ngram-int8']['source_prediction_preserved'].any())
            self.assertEqual(file_hash(checkpoint/'head.npz'),training['head_sha256'])
            result = subprocess.run([sys.executable,'-m','tools.export_news_classifier','--training',str(parent),
                '--output',str(parent/'cli-serving'),'--seed','3','--backends','source-mmap,ngram-bounded,ngram-int8',
                '--native-library',str(self.library)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with TextClassifier(parent/'cli-serving/ngram-int8',native_library=self.library) as model:
                self.assertEqual(model.predict(['ab'])['predictions'].tolist(),[0])

    def test_metrics_and_validation_only_budget_selection(self):
        from tools.benchmark_news_classifier import metrics, select
        result = metrics(np.asarray([0,1,2,3]),np.asarray([0,1,1,3]),np.asarray([.1,.2,.3,.4]))
        self.assertEqual(result['accuracy'],.75)
        self.assertAlmostEqual(result['macro_f1'],2/3)
        self.assertEqual(metrics(np.asarray([0]),np.asarray([0]),np.asarray([0.]),[False])['documents'],0)
        def candidate(mib,rss,accuracy,ce,rate):
            return {'complete_model_bytes':mib*(1<<20),'peak_rss_bytes':rss*(1<<20),
                    'metrics':{'accuracy':accuracy,'cross_entropy':ce},
                    'timings':{'32':{'documents_per_second':rate}}}
        records = {'source-mmap':candidate(400,400,.9,.3,100),
                   'ngram':candidate(50,200,.896,.301,120),'fp16':candidate(80,200,.899,.301,150),
                   'bad-small':candidate(2,10,.85,.3,1000)}
        choices = select(records)
        self.assertIsNone(next(c for c in choices if c['budget_kind']=='artifact' and c['budget_mib']==16)['selected'])
        self.assertEqual(next(c for c in choices if c['budget_kind']=='artifact' and c['budget_mib']==64)['selected'],'ngram')
        self.assertEqual(next(c for c in choices if c['budget_kind']=='observed_rss' and c['budget_mib']==256)['selected'],'fp16')
        self.assertFalse(any('bad-small' in c['eligible'] for c in choices))


if __name__ == '__main__':
    unittest.main()
