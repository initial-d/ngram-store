import json
import tempfile
import unittest
from pathlib import Path
from tools.counterfact_data import load_records,build_sequences,batch_indices,case_metrics,summarize


class Tokenizer:
    eos_token_id=999
    def __call__(self,text,add_special_tokens=False):
        return type('Encoding',(),{'input_ids':[ord(x) for x in text]})()


class CounterFactDataTest(unittest.TestCase):
    def fixture(self):
        return [{'case_id':2,'requested_rewrite':{'subject':'Alice','relation_id':'P1','prompt':'{} lives in',
            'target_new':{'str':'Paris'},'target_true':{'str':'Rome'}},
            'paraphrase_prompts':['The home of Alice is'],'neighborhood_prompts':['Bob lives in']}]

    def test_training_contains_only_rewrite_target_and_eos(self):
        sequences,train,cases=build_sequences(self.fixture(),Tokenizer())
        self.assertEqual(len(train),1)
        s=sequences[train[0]]
        self.assertEqual(s['ids'][:-1],[ord(x) for x in 'Alice lives in Paris'])
        self.assertEqual(s['ids'][-1],999)
        self.assertEqual(s['target_start'],len('Alice lives in'))
        for kind in ('paraphrase','neighborhood'):
            self.assertTrue(all(p['new'] not in train and p['true'] not in train for p in cases[0]['groups'][kind]))
        batches=batch_indices(list(range(len(sequences))),sequences,2)
        self.assertEqual(sorted(i for b in batches for i in b),list(range(len(sequences))))
        self.assertTrue(all(len({len(sequences[i]['ids']) for i in b})==1 for b in batches))

    def test_duplicate_batch_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'records.json';p.write_text(json.dumps(self.fixture()*2))
            with self.assertRaises(ValueError):load_records(p,expected_sha=None)

    def test_official_success_direction_and_case_macro_average(self):
        sequences,_,cases=build_sequences(self.fixture(),Tokenizer())
        scores={str(i):{'nll':2.,'correct':False} for i in range(len(sequences))}
        for group,pairs in cases[0]['groups'].items():
            for p in pairs:
                winner=p['true'] if group=='neighborhood' else p['new']
                scores[str(winner)]={'nll':1.,'correct':True}
        result=summarize(case_metrics(cases,scores))
        self.assertEqual(result['harmonic_score'],1.)
        self.assertEqual(result['neighborhood']['success'],1.)
