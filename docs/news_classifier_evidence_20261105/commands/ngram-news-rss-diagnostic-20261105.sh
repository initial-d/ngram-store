#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
/llm-align/duyimin/miniconda3/envs/gemma4/bin/python - <<'PY'
import json, resource, subprocess, sys
from pathlib import Path
code = "import resource;from pathlib import Path;print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);print(next(l for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmHWM:')))"
observations = []
for mib in (0,256):
    allocation = bytearray(mib*(1<<20))
    observations.append({'parent_allocation_mib':mib,'parent_ru_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'child':subprocess.check_output([sys.executable,'-c',code],text=True).splitlines()})
    del allocation
path = Path('/llm-align/duyimin/ngram-news-app-20261105/public-metadata/rss-diagnostic.json')
path.write_text(json.dumps(observations,indent=2)+'\n')
print(path.read_text())
PY
