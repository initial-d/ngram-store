"""Reject scientifically misleading aggregation and inconsistent sufficient stats."""
import unittest

from tools.verify_memory_application import gate, metrics, paired


class EvidenceArithmeticTests(unittest.TestCase):
    def test_shared_statistics_preserve_original_json_and_signed_zero(self):
        try:
            from tools.publish_memory_application import share_statistics,expand_statistics
        except ImportError:
            self.skipTest('optional NumPy required')
        import json,hashlib
        blocks={'all':{'candidate_loss':[1.,2.,-0.], 'baseline_loss':[1.,2.,0.], 'count':[1,2,0]},
                'known':{'candidate_loss':[1.,2.,-0.], 'baseline_loss':[1.,2.,0.], 'count':[1,2,0]}}
        arrays={};refs=share_statistics(blocks,arrays)
        self.assertEqual(len(arrays),3)
        self.assertNotEqual(refs['all']['candidate_loss'],refs['all']['baseline_loss'])
        first=json.dumps(blocks,indent=2,sort_keys=True).encode()
        second=json.dumps(expand_statistics(refs,arrays),indent=2,sort_keys=True).encode()
        self.assertEqual(hashlib.sha256(first).digest(),hashlib.sha256(second).digest())

    def test_official_scores_weight_cases_not_prompt_count(self):
        cases = []
        for index, successes in enumerate(([True], [False] * 9)):
            groups = {}
            for group in ('rewrite', 'paraphrase', 'neighborhood'):
                groups[group] = [{'success': win, 'correct': win,
                    'target_new': 2. if win == (group == 'neighborhood') else 1.,
                    'target_true': 1. if win == (group == 'neighborhood') else 2.} for win in successes]
            cases.append({'case_id': index, 'groups': groups})
        result = metrics(cases)
        self.assertEqual(result['rewrite']['prompts'], 10)
        self.assertEqual(result['rewrite']['success'], .5)
        self.assertEqual(result['harmonic_score'], .5)
        cases[0]['groups']['neighborhood'][0]['success'] = False
        with self.assertRaises(AssertionError):
            metrics(cases)

    def test_an_oov_failure_cannot_hide_inside_pooled_quality(self):
        quality = {s: {'pairs': 10, 'loss_relative_change': 0., 'discrimination_change': 0.}
                   for s in ('all', 'oov_nonempty')}
        self.assertTrue(gate(quality, .001))
        quality['oov_nonempty']['loss_relative_change'] = .002
        self.assertFalse(gate(quality, .001))
        quality['oov_nonempty']['loss_relative_change'] = 0.
        quality['oov_nonempty']['discrimination_change'] = -.006
        self.assertFalse(gate(quality, .001))

    def test_subgroup_stats_must_partition_all_context_pairs(self):
        empty = {'block_ids': [], 'count': [], 'candidate_loss': [], 'baseline_loss': [],
                 'candidate_discrimination': [], 'baseline_discrimination': []}
        block = {'block_ids': [0], 'count': [2], 'candidate_loss': [2.], 'baseline_loss': [2.],
                 'candidate_discrimination': [1.], 'baseline_discrimination': [1.]}
        full = {'pairs': 2, 'sgns_loss_per_pair': 1., 'baseline_sgns_loss_per_pair': 1.,
                'context_discrimination': .5, 'baseline_context_discrimination': .5,
                'loss_relative_change': 0., 'discrimination_change': 0.}
        absent = {'pairs': 0, 'loss_relative_change': None, 'discrimination_change': None}
        blocks = {'all': block, 'known': block, 'oov_nonempty': empty, 'oov_empty': empty}
        quality = {'all': full, 'known': full, 'oov_nonempty': absent, 'oov_empty': absent}
        paired(blocks, quality, {'pairs': 2, 'tokens': 10})
        blocks['oov_nonempty'] = block; quality['oov_nonempty'] = full
        with self.assertRaises(AssertionError):
            paired(blocks, quality, {'pairs': 2, 'tokens': 10})

    def test_pairing_cancels_case_and_seed_variation(self):
        try:
            import numpy as np
            from tools.analyze_memory_application import hierarchical_paired
        except ImportError:
            self.skipTest('optional NumPy required')
        rng = np.random.default_rng(123)
        sources = [rng.uniform(.1, .9, (40, 3)) for _ in range(3)]
        result = hierarchical_paired(sources, [s.copy() for s in sources], replicates=50)
        self.assertEqual(result['observed_candidate_minus_source'], [0.] * 4)
        self.assertEqual(result['paired_95_percent_intervals'], [[0., 0.]] * 4)
        self.assertTrue(all(v == [0.] * 4 for v in result['samples']))

    def test_real_batches_preserve_length_and_drop_duplicate_prompts(self):
        try:
            from tools.benchmark_llm_codec_serving import batches_for
            from tools.tree_node_quality import write, digest
        except ImportError:
            self.skipTest('optional NumPy required')
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);common=root/'common';prepared=root/'prepared'
            common.mkdir();prepared.mkdir()
            sequences=[{'ids':[i,100,200],'target_start':2} for i in range(8)]
            sequences.extend([dict(sequences[0]),{'ids':[1,2,3,200],'target_start':3}])
            cases=[{'case_id':i,'split':'test','groups':{'rewrite':[{'new':i}], 'paraphrase':[], 'neighborhood':[]}}
                   for i in (0,8,9,1,2,3,4,5,6,7)]
            write(prepared/'sequences.json',{'sequences':sequences,'cases':cases})
            write(prepared/'prepare.json',{'payload_sha256':digest(prepared/'sequences.json')})
            write(common/'metadata.json',{'prepared':str(prepared)})
            actual=batches_for(common,'test',1,8)
            self.assertEqual(len(actual),1)
            self.assertEqual([r['case_id'] for r in actual[0]],list(range(8)))
            self.assertEqual({len(r['ids']) for r in actual[0]},{2})
            self.assertEqual(len({tuple(r['ids']) for r in actual[0]}),8)

    def test_per_device_queue_keeps_every_request_and_resumes_verified_outputs(self):
        try:
            from tools.finalize_memory_application import gpu_parallel
            from tools.tree_node_quality import write, load, digest
        except ImportError:
            self.skipTest('optional NumPy required')
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frontier=root/'frontier';frontier.mkdir()
            write(frontier/'selection.json',{'fixture':'frozen'})
            names=[f'fixture-{i}' for i in range(6)]
            amendment=root/'addendum.json';affinity=root/'affinity.json';ready=root/'ready.json'
            write(amendment,{'device_assignment':dict(zip(names,range(6)))})
            write(affinity,{name:[i] for i,name in enumerate(names)})
            write(ready,{'status':'quality-complete','fixture':True})
            requests=[]
            for name in names:
                job=root/(name+'.json');write(job,{'fixture':True})
                requests.extend({'model':name,'name':'raw','job':str(job),'mode':'codec','cache':0,'batch':batch} for batch in (1,8))
            args=SimpleNamespace(output=root/'timers',frontier=frontier,ready=ready,gpu_addendum=amendment,
                                 gpu_affinity=affinity,phase='retime-llm-gpu',upstream=root,requests=32)
            def worker(command,env,stdout,stderr):
                job=Path(command[command.index('--job')+1]);output=Path(command[command.index('--output')+1])
                cpus=list(map(int,command[command.index('--cpu-list')+1].split(',')))
                self.assertEqual(cpus,[int(env['CUDA_VISIBLE_DEVICES'])])
                write(output,{'status':'passed','job_sha256':digest(job),'cpu_affinity':cpus,'fixture':True})
                return SimpleNamespace(returncode=0)
            with patch('os.sched_getaffinity',return_value=set(range(6)),create=True), \
                 patch('tools.finalize_memory_application.environment',return_value={'fixture':True}), \
                 patch('tools.finalize_memory_application.subprocess.run',side_effect=worker) as execution:
                gpu_parallel(args,requests,load(ready))
                self.assertEqual(execution.call_count,12)
            first=load(args.output/'report.json')
            self.assertEqual(first['status'],'complete')
            self.assertEqual(len(first['complete']),12)
            with patch('os.sched_getaffinity',return_value=set(range(6)),create=True), \
                 patch('tools.finalize_memory_application.environment',return_value={'fixture':True}), \
                 patch('tools.finalize_memory_application.subprocess.run') as execution:
                gpu_parallel(args,requests,load(ready))
                execution.assert_not_called()
            second=load(args.output/'report.json')
            self.assertEqual(second['execution_schedule_sha256'],first['execution_schedule_sha256'])
            self.assertEqual(second['requests'],first['requests'])


if __name__ == '__main__':
    unittest.main()
