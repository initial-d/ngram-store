"""External interruptions are retried without accepting contaminated timings."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.gpu_job_retry import run_gpu_job


class GpuJobRetryTests(unittest.TestCase):
    def test_competing_job_is_discarded_hashed_and_repeated_with_same_command(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'job.json'; log=Path(temp)/'job.log'; command=['worker','--requests','32']
            env={'CUDA_VISIBLE_DEVICES':'7'}; seen=[]
            def run(cmd, env, stdout, stderr):
                seen.append((cmd,dict(env)))
                if len(seen)==1:
                    output.write_text(json.dumps({'status':'failed','error':'GPU competition appeared during serving','samples':['contaminated']}))
                    return SimpleNamespace(returncode=1)
                output.write_text(json.dumps({'status':'passed','samples':['fresh']})); return SimpleNamespace(returncode=0)
            with patch('tools.gpu_job_retry.subprocess.run',side_effect=run), patch('tools.gpu_job_retry.wait_idle') as idle:
                self.assertEqual(run_gpu_job(command,env=env,output=output,log=log).returncode,0)
                idle.assert_called_once()
            self.assertEqual(seen,[(command,env)]*2)
            self.assertEqual(json.loads(output.read_text())['samples'],['fresh'])
            retained=list(Path(temp).glob('job.interrupted-*.json'))
            self.assertEqual(len(retained),1)
            self.assertEqual(json.loads(retained[0].read_text())['samples'],['contaminated'])
            event=json.loads((Path(temp)/'job.interruptions.jsonl').read_text())
            self.assertEqual(event['output'],str(retained[0]));self.assertEqual(len(event['sha256']),64)

    def test_algorithm_failure_does_not_turn_into_an_automatic_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'job.json';log=Path(temp)/'job.log'
            def run(*args,**kwargs):
                output.write_text(json.dumps({'status':'failed','error':'decoded row bytes changed'}))
                return SimpleNamespace(returncode=1)
            with patch('tools.gpu_job_retry.subprocess.run',side_effect=run) as runner, patch('tools.gpu_job_retry.wait_idle') as idle:
                self.assertEqual(run_gpu_job(['worker'],env={'CUDA_VISIBLE_DEVICES':'0'},output=output,log=log).returncode,1)
                runner.assert_called_once();idle.assert_not_called()
                with self.assertRaisesRegex(ValueError,'needs diagnosis'):
                    run_gpu_job(['worker'],env={'CUDA_VISIBLE_DEVICES':'0'},output=output,log=log)
            self.assertEqual(json.loads(output.read_text())['error'],'decoded row bytes changed')

    def test_a_passed_output_is_never_overwritten_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'job.json';output.write_text('{"status":"passed","samples":[1,2,3]}')
            original=output.read_bytes()
            with patch('tools.gpu_job_retry.subprocess.run') as runner:
                self.assertEqual(run_gpu_job(['worker'],env={},output=output,log=Path(temp)/'log').returncode,0)
                runner.assert_not_called()
            self.assertEqual(output.read_bytes(),original)

    def test_resuming_an_interrupted_output_preserves_it(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'job.json';log=Path(temp)/'job.log'
            output.write_text('{"status":"failed","error":"GPU competition before model loading"}')
            def run(*args,**kwargs):
                self.assertFalse(output.exists())
                output.write_text('{"status":"passed"}');return SimpleNamespace(returncode=0)
            with patch('tools.gpu_job_retry.subprocess.run',side_effect=run):
                self.assertEqual(run_gpu_job(['worker'],env={},output=output,log=log).returncode,0)
            self.assertEqual(len(list(Path(temp).glob('job.interrupted-*.json'))),1)


if __name__=='__main__':unittest.main()
