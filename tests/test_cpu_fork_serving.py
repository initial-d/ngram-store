"""Fork IPC retains fresh outputs and the original phase/core/job domain."""
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.cpu_fork_serving import encode_args, socket_path, submit, validate


class ForkServingTests(unittest.TestCase):
    def test_immutable_job_phase_and_cpu_assignment_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/'plan.json';job=root/'long-frontier/model/jobs/job.json'
            job.parent.mkdir(parents=True);job.write_text('{}')
            plan.write_text(json.dumps({'measurement_host':socket.gethostname(),'lanes':{'0':{'cpus':[0]}},
                'model_lanes':{'retime-long':{'model':'0'}}}))
            args=SimpleNamespace(phase='retime-long',plan=plan,job=job,output=root/'retime-long/00000.json',
                lane='0',cache=0,requests=32,upstream=None)
            actual,_=validate({'args':encode_args(args)},plan,'retime-long')
            self.assertEqual(actual.job,job)
            for field,value in [('requests',16),('output',root/'elsewhere/00000.json'),('phase','retime-llm-cpu')]:
                altered=encode_args(args);altered[field]=str(value) if isinstance(value,Path) else value
                with self.subTest(field=field),self.assertRaises(ValueError):validate({'args':altered},plan,'retime-long')
            args.output.parent.mkdir();args.output.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Fresh'):validate({'args':encode_args(args)},plan,'retime-long')

    def test_absent_service_preserves_the_existing_direct_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(plan=Path(temp)/'plan.json',phase='retime-long')
            self.assertFalse(submit(args))
            self.assertEqual(socket_path(args.plan,args.phase).parent,Path(temp))


if __name__=='__main__':unittest.main()
