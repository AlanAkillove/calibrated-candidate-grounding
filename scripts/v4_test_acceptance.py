"""Run complete repository tests into V4 and fingerprint actual tested bytes."""
from __future__ import annotations
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
from ccg.v4.freeze import sha256_file

def main():
    import pytest
    out = ROOT / 'results/v4_targeted_strengthening/acceptance'
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / 'full_tests.json'
    if record_path.exists():
        raise FileExistsError('Preserve prior test evidence; create an explicit correction version')
    paths = sorted(set((ROOT/'src').rglob('*.py')) | set((ROOT/'tests').rglob('*.py')) | set((ROOT/'scripts').glob('*.py')))
    before = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    started = datetime.now(timezone.utc).isoformat()
    junit = out / 'full_tests.xml'
    code = pytest.main([str(ROOT/'tests'),'-o','addopts=','-q',f'--junitxml={junit}'])
    changed = [p for p,h in before.items() if not (ROOT/p).is_file() or sha256_file(ROOT/p) != h]
    new_paths = sorted(set(p.relative_to(ROOT).as_posix() for p in
        set((ROOT/'src').rglob('*.py')) | set((ROOT/'tests').rglob('*.py')) | set((ROOT/'scripts').glob('*.py'))) - set(before))
    counts = dict(tests=0, errors=0, failures=0, skipped=0)
    if junit.exists():
        tree = ET.parse(junit).getroot()
        suites = [tree] if tree.tag == 'testsuite' else list(tree.findall('testsuite'))
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.get(key,'0'))
    record = dict(status='PASS' if int(code)==0 and not changed and not new_paths else 'FAIL_OR_SOURCE_CHANGED',
        actual_invocation=[sys.executable,'-B','scripts/v4_test_acceptance.py'],
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        counts=counts, exit_code=int(code), changed_during_tests=changed, new_during_tests=new_paths,
        tested_source_inputs=before, junit_path=junit.relative_to(ROOT).as_posix(),
        junit_sha256=sha256_file(junit) if junit.exists() else None,
        limitation='Tests do not replace actual source anchors, identity or frozen statistical distribution acceptance.')
    record_path.write_text(json.dumps(record,indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('status','counts','exit_code','changed_during_tests','new_during_tests')}), flush=True)
    return 0 if record['status']=='PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
