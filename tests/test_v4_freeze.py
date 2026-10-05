from pathlib import Path
import hashlib
import json
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ccg.v4.freeze import verify_freeze

def make_freeze(tmp_path):
    file = tmp_path / 'input.txt'
    file.write_bytes(b'fixed input')
    target = tmp_path / 'results/v4_targeted_strengthening/p0_freeze.json'
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(dict(stage='p0', schema='v4-stage-freeze-v1',
        files=[dict(path='input.txt', size_bytes=11, sha256=hashlib.sha256(file.read_bytes()).hexdigest())])), encoding='utf-8')
    return file, target

def test_verify_existing_stage(tmp_path):
    make_freeze(tmp_path)
    assert verify_freeze('p0', root=tmp_path)['files'] == 1

def test_changed_bytes_refused_even_same_length(tmp_path):
    file, _ = make_freeze(tmp_path)
    file.write_bytes(b'other input')
    with pytest.raises(AssertionError, match='changed'):
        verify_freeze('p0', root=tmp_path)

def test_repeated_input_refused(tmp_path):
    _, target = make_freeze(tmp_path)
    payload = json.loads(target.read_text())
    payload['files'].append(payload['files'][0])
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='duplicate'):
        verify_freeze('p0', root=tmp_path)

def test_outside_input_refused(tmp_path):
    _, target = make_freeze(tmp_path)
    payload = json.loads(target.read_text())
    payload['files'][0]['path'] = '../outside'
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='unsafe'):
        verify_freeze('p0', root=tmp_path)
