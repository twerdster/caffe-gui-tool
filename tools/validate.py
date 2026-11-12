"""Run the complete local validation in isolated Blender processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender', default=shutil.which('blender'))
    parser.add_argument('--python', default=sys.executable, help='Python with requirements-test.txt installed')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/validation')
    parser.add_argument('--skip-protobuf', action='store_true', help='Explicitly omit schema validation')
    args = parser.parse_args()
    if not args.blender:
        parser.error('Blender is required; provide --blender /path/to/blender')
    if not args.skip_protobuf and not shutil.which('protoc'):
        parser.error('protoc is required; install it or explicitly use --skip-protobuf')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'schema_validation': not args.skip_protobuf, 'suites': {}}

    def run(label, command, environment=None):
        print(f'Running {label}...', flush=True)
        start = time.monotonic()
        try:
            process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                     text=True, timeout=600)
        except subprocess.TimeoutExpired as error:
            def decode(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
            (output / (label + '.log')).write_text(decode(error.stdout) + decode(error.stderr))
            report['suites'][label] = {'status': 'timeout', 'seconds': round(time.monotonic() - start, 3)}
            raise RuntimeError(f'{label} timed out; see {output / (label + ".log")}') from error
        (output / (label + '.log')).write_text(process.stdout + process.stderr)
        entry = {'exit_code': process.returncode, 'seconds': round(time.monotonic() - start, 3)}
        for line in process.stdout.splitlines():
            if line.startswith('VALIDATION_JSON='):
                entry['results'] = json.loads(line.split('=', 1)[1])
        report['suites'][label] = entry
        if process.returncode:
            raise RuntimeError(f'{label} failed; see {output / (label + ".log")}')
        print(f'Passed {label} ({entry["seconds"]}s)', flush=True)

    try:
        run('runtime_dependencies', [args.python, '-c',
            'import numpy, cv2, pytest, PIL; '
            'assert hasattr(cv2.dnn, "readNetFromCaffe"), "OpenCV 4 Caffe importer is required"'])
        with tempfile.TemporaryDirectory(prefix='caffe-gui-validation-') as temporary:
            environment = os.environ.copy()
            for setting, directory in [('BLENDER_USER_CONFIG', 'config'),
                                       ('BLENDER_USER_SCRIPTS', 'scripts'),
                                       ('BLENDER_USER_EXTENSIONS', 'extensions'),
                                       ('BLENDER_USER_DATAFILES', 'datafiles')]:
                path = Path(temporary) / directory
                path.mkdir()
                environment[setting] = str(path)
            environment['PYTHONPATH'] = ''
            base = [args.blender, '--background', '--factory-startup', '--python-exit-code', '1']
            suites = ['blender_test', 'blender_graph_regressions', 'blender_export_regressions', 'blender_integration']
            for label in suites:
                command = base + ['--python', str(ROOT / 'tests' / (label + '.py'))]
                if label == 'blender_test' and not args.skip_protobuf:
                    command += ['--', '--caffe-proto', str(ROOT / 'tests/schema/caffe.proto')]
                run(label, command, environment)
            run('build_addon', [sys.executable, str(ROOT / 'tools/build_addon.py')])
            archive = ROOT / 'dist/caffe-gui-tool-2.0.0.zip'
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            run('build_reproducibility', [sys.executable, str(ROOT / 'tools/build_addon.py')])
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
            with zipfile.ZipFile(archive) as package:
                assert package.testzip() is None
                assert all(not name.endswith(('.blend', '.pyc', '.caffemodel', '.png', '.h5')) for name in package.namelist())
            report['package'] = {'bytes': archive.stat().st_size, 'sha256': digest, 'reproducible': True}
            run('blender_install', base + ['--python', str(ROOT / 'tests/blender_install.py')], environment)
            fixtures = output / 'runtime-fixtures'
            run('blender_runtime_fixtures', base + ['--python', str(ROOT / 'tests/blender_runtime_fixtures.py'),
                                                  '--', '--output', str(fixtures)], environment)
            runtime_report = output / 'runtime.json'
            runtime_report.unlink(missing_ok=True)
            runtime_environment = environment | {'CAFFE_RUNTIME_FIXTURES': str(fixtures),
                'CAFFE_RUNTIME_REPORT': str(runtime_report)}
            run('runtime', [args.python, '-m', 'pytest', '-q', 'tests/test_caffe_runtime.py'], runtime_environment)
            report['runtime'] = json.loads(runtime_report.read_text())
            assert report['runtime']['public_image_cases'] == 487, 'Public-image coverage was incomplete'
            assert report['runtime']['synthetic_cases'] >= 23, 'Synthetic coverage was incomplete'
            assert report['runtime']['repeat_identical'] >= 510, 'Repeatability coverage was incomplete'
            run('harness_regressions', [args.python, '-m', 'pytest', '-q', 'tests/test_validation_harness.py'])
            report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'All checks passed. Report: {output / "summary.json"}')


if __name__ == '__main__':
    main()
