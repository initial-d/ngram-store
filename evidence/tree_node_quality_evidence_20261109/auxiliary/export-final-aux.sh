#!/bin/bash
set -eu
TASK=${RUN_ROOT}
while ! test -f "$TASK/publication-complete" || ! test -f "$TASK/rows-publication-complete" || ! test -f "$TASK/input-audit-complete"; do sleep 10; done
${ENV_ROOT}/bin/python "$TASK/export-final-aux.py"
