"""Prevent old numerical reports from certifying a skipped current run."""
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import validate


@pytest.mark.parametrize('fresh_report', [False, True])
def test_runtime_evidence_must_come_from_current_invocation(tmp_path, monkeypatch, fresh_report):
    root = tmp_path / 'repo'
    archive = root / 'dist/caffe-gui-tool-2.0.0.zip'
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, 'w') as package:
        package.writestr('neural_operator_blender/__init__.py', '# fixture\n')
    output = root / 'results'
    output.mkdir()
    previous = {'public_image_cases': 487, 'synthetic_cases': 23,
                'repeat_identical': 510, 'stale_evidence': True}
    (output / 'runtime.json').write_text(json.dumps(previous))
    monkeypatch.setattr(validate, 'ROOT', root)
    monkeypatch.setattr(sys, 'argv', ['validate.py', '--blender', '/fake/blender',
                                    '--python', '/fake/python', '--skip-protobuf',
                                    '--output', str(output)])

    def process(command, **kwargs):
        if 'tests/test_caffe_runtime.py' in command:
            destination = Path(kwargs['env']['CAFFE_RUNTIME_REPORT'])
            assert not destination.exists(), 'Old evidence must be removed before pytest starts'
            if fresh_report:
                destination.write_text(json.dumps(previous | {'stale_evidence': False}))
            # Model pytest exiting successfully after numerical tests skipped.
            return subprocess.CompletedProcess(command, 0, '1 passed, 526 skipped\n', '')
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(subprocess, 'run', process)
    if fresh_report:
        validate.main()
        report = json.loads((output / 'summary.json').read_text())
        assert report['status'] == 'passed'
        assert report['runtime']['stale_evidence'] is False
    else:
        with pytest.raises(FileNotFoundError):
            validate.main()
        report = json.loads((output / 'summary.json').read_text())
        assert report['status'] == 'failed'
        assert 'runtime' not in report
