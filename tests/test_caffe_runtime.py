"""Numerical Blender-export -> Caffe -> OpenCV CPU tests against NumPy.

Run with the optional runtime dependencies installed and Blender available:
  BLENDER_EXECUTABLE=/path/to/blender python -m pytest tests/test_caffe_runtime.py
CAFFE_RUNTIME_FIXTURES can point at previously generated Blender fixtures.
CAFFE_RUNTIME_REPORT writes a small numerical validation record as JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_caffe import CaffeRunner
from tools import run_caffe


@pytest.fixture(scope='session')
def exported(tmp_path_factory):
    cv2 = pytest.importorskip('cv2', reason='Optional Caffe runtime tests need OpenCV 4')
    external = os.environ.get('CAFFE_RUNTIME_FIXTURES')
    destination = Path(external) if external else tmp_path_factory.mktemp('blender-caffe-runtime')
    if not external:
        candidates = [os.environ.get('BLENDER_EXECUTABLE'), shutil.which('blender'),
                      '/Volumes/Blender/Blender.app/Contents/MacOS/Blender',
                      '/Applications/Blender.app/Contents/MacOS/Blender']
        blender = next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)
        if not blender:
            pytest.skip('Numerical export tests require Blender; set BLENDER_EXECUTABLE')
        process = subprocess.run([
            blender, '--background', '--factory-startup', '--python-exit-code', '1',
            '--python', str(ROOT / 'tests/blender_runtime_fixtures.py'),
            '--', '--output', str(destination),
        ], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert process.returncode == 0, process.stdout + process.stderr
    fixtures = {}
    for channels in (1, 3):
        directory = destination / f'channels-{channels}'
        manifest = json.loads((directory / 'manifest.json').read_text())
        fixtures[channels] = {
            'directory': directory, 'manifest': manifest,
            'runner': CaffeRunner(directory / 'runtime_deploy.prototxt',
                                  directory / 'coefficients.caffemodel'),
        }
    report = {'opencv': cv2.__version__, 'numpy': np.__version__,
              'blender': fixtures[1]['manifest']['blender'],
              'synthetic_cases': 0, 'public_image_cases': 0, 'layer_comparisons': 0,
              'rejected_grouped_geometries': [],
              'repeat_identical': 0, 'max_absolute_error': 0.0, 'worst_case': None,
              'trained': False, 'reference': 'Independent NumPy grouped correlation and arithmetic'}
    yield fixtures, report
    if os.environ.get('CAFFE_RUNTIME_REPORT'):
        path = Path(os.environ['CAFFE_RUNTIME_REPORT'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')


def reference(data, manifest):
    """Depthwise cross-correlation with explicit independent NumPy indexing."""
    values = np.asarray(data, dtype=np.float32)
    padded = np.pad(values, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    batch, channels, height, width = values.shape
    correlation = np.zeros_like(values)
    kernels = np.asarray(manifest['kernel'], dtype=np.float32).reshape(channels, 3, 3)
    for row in range(3):
        for column in range(3):
            correlation += padded[:, :, row:row + height, column:column + width] * kernels[None, :, row, column, None, None]
    correlation += np.asarray(manifest['bias'], dtype=np.float32)[None, :, None, None]
    activation = np.where(correlation >= 0, correlation, correlation * np.float32(0.125))
    power = np.power(activation * np.float32(0.125) + np.float32(2.0), np.float32(1.5))
    summation = power * np.float32(0.75) + values * np.float32(-0.125)
    product = summation * power
    return dict(zip(manifest['layers'], (correlation, activation, power, summation, product)))


def compare(data, fixture, report, name, all_layers):
    reference_values = reference(data, fixture['manifest'])
    layers = fixture['manifest']['layers'] if all_layers else ['product']
    for layer in layers:
        actual = fixture['runner'].predict(data, output_layer=layer)
        expected = reference_values[layer]
        assert actual.shape == expected.shape, name
        assert np.isfinite(actual).all(), name
        np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=3e-5,
                                   err_msg=f'{name}: {layer}')
        error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
        report['layer_comparisons'] += 1
        if error > report['max_absolute_error']:
            report['max_absolute_error'], report['worst_case'] = error, f'{name}:{layer}'
    repeated = fixture['runner'].predict(data)
    np.testing.assert_array_equal(repeated, fixture['runner'].predict(data), err_msg=f'{name}: repeat drift')
    np.testing.assert_allclose(repeated, reference_values['product'], rtol=4e-6, atol=3e-5)
    report['repeat_identical'] += 1


PATTERNS = ['zeros', 'positive_constant', 'negative_constant', 'horizontal_ramp',
            'vertical_ramp', 'center_impulse', 'corner_impulse', 'checkerboard', 'noise']


def pattern_data(pattern, channels, height=11, width=17, batch=1):
    rows, columns = np.mgrid[:height, :width]
    if pattern == 'zeros':
        image = np.zeros((height, width), dtype=np.float32)
    elif pattern.endswith('constant'):
        image = np.full((height, width), 1 if pattern.startswith('positive') else -1, dtype=np.float32)
    elif pattern == 'horizontal_ramp':
        image = np.broadcast_to(np.linspace(-1, 1, width), (height, width)).copy()
    elif pattern == 'vertical_ramp':
        image = np.broadcast_to(np.linspace(-1, 1, height)[:, None], (height, width)).copy()
    elif pattern.endswith('impulse'):
        image = np.zeros((height, width), dtype=np.float32)
        image[(height // 2, width // 2) if pattern.startswith('center') else (0, 0)] = 1
    elif pattern == 'checkerboard':
        image = ((rows + columns) % 2) * 2 - 1
    else:
        image = np.random.default_rng(2017).normal(0, 0.5, (height, width))
    result = np.stack([image * (1 - channel * 0.2) + channel * 0.1 for channel in range(channels)])
    return np.stack([result if index == 0 else result[:, ::-1, :] for index in range(batch)]).astype(np.float32)


@pytest.mark.parametrize('channels', [1, 3])
@pytest.mark.parametrize('pattern', PATTERNS)
def test_synthetic_exported_math(exported, channels, pattern):
    fixtures, report = exported
    compare(pattern_data(pattern, channels), fixtures[channels], report,
            f'synthetic/{channels}/{pattern}', all_layers=True)
    report['synthetic_cases'] += 1


@pytest.mark.parametrize('channels,height,width,batch', [(1, 1, 1, 1), (3, 3, 9, 1),
                                                      (1, 19, 1, 1), (3, 7, 23, 2),
                                                      (1, 24, 13, 3)])
def test_dynamic_non_square_and_batch_shapes(exported, channels, height, width, batch):
    fixtures, report = exported
    data = pattern_data('noise', channels, height, width, batch)
    # A noncontiguous input exercises the runtime array adapter.
    data = data[:, :, :, ::-1]
    assert not data.flags.c_contiguous or width == 1
    compare(data, fixtures[channels], report, f'shape/{data.shape}', all_layers=True)
    report['synthetic_cases'] += 1


@pytest.mark.parametrize('height,width', [(1, 9), (11, 1), (2, 13), (7, 2)])
def test_unsafe_thin_grouped_convolution_rejected(exported, height, width):
    fixtures, report = exported
    with pytest.raises(ValueError, match='at least 3 by 3'):
        fixtures[3]['runner'].predict(pattern_data('noise', 3, height, width))
    report['rejected_grouped_geometries'].append([1, 3, height, width])


def test_grouped_geometry_guard_checks_after_pooling(exported, tmp_path):
    fixtures, _ = exported
    deploy = tmp_path / 'downsampled.prototxt'
    deploy.write_text('input: "data" input_shape { dim:1 dim:3 dim:4 dim:4 }\n'
                      'layer { name:"downsample" type:"Pooling" bottom:"data" top:"small" '
                      'pooling_param { pool:MAX kernel_size:2 stride:2 } }\n'
                      'layer { name:"correlation" type:"Convolution" bottom:"small" top:"output" '
                      'convolution_param { num_output:3 group:3 kernel_size:3 pad:1 bias_term:true } }')
    runner = CaffeRunner(deploy, fixtures[3]['directory'] / 'coefficients.caffemodel')
    with pytest.raises(ValueError, match='its input is 2 by 2'):
        runner.predict(np.ones((1, 3, 4, 4), dtype=np.float32))


DATA_ROOT = ROOT / 'patch_beltrami/Beltrami'
PUBLIC_IMAGES = sorted(path for directory in ('68_IMGS', '400_IMGS', 'BM3D_images')
                       for path in (DATA_ROOT / directory).glob('*')
                       if path.suffix.lower() in {'.png', '.tif', '.tiff', '.jpg'})


def test_public_image_corpus_complete():
    expected = {'68_IMGS': 68, '400_IMGS': 400, 'BM3D_images': 19}
    actual = {directory: sum(path.parent.name == directory for path in PUBLIC_IMAGES)
              for directory in expected}
    assert actual == expected, 'The full numerical suite requires all 487 supplied public images'


@pytest.mark.parametrize('path', PUBLIC_IMAGES, ids=lambda path: str(path.relative_to(DATA_ROOT)))
def test_public_images_at_native_resolution(exported, path):
    fixtures, report = exported
    with Image.open(path) as image:
        channels = 3 if len(image.getbands()) >= 3 else 1
        pixels = np.asarray(image.convert('RGB' if channels == 3 else 'L'), dtype=np.float32)
    data = pixels.transpose(2, 0, 1)[None] if channels == 3 else pixels[None, None]
    data = data / np.float32(127.5) - np.float32(1.0)
    compare(data, fixtures[channels], report, str(path.relative_to(ROOT)), all_layers=False)
    report['public_image_cases'] += 1


@pytest.mark.parametrize('value', [np.ones((3, 5)), np.empty((1, 1, 0, 4)),
                                  np.full((1, 1, 3, 4), np.nan),
                                  np.full((1, 1, 3, 4), np.inf),
                                  np.full((1, 1, 3, 4), 1e300),
                                  np.ones((1, 1, 3, 4), dtype=np.complex64),
                                  np.full((1, 1, 3, 4), 'text')])
def test_invalid_arrays_rejected(exported, value):
    fixtures, _ = exported
    with pytest.raises(ValueError):
        fixtures[1]['runner'].predict(value)


def test_missing_weights_and_historical_padding_rejected(exported, tmp_path):
    fixtures, _ = exported
    with pytest.raises(ValueError, match='explicit .caffemodel'):
        CaffeRunner(fixtures[1]['directory'] / 'runtime_deploy.prototxt')
    unsupported = tmp_path / 'research.prototxt'
    unsupported.write_text('layer { name: "pad" type: "Padding" }')
    with pytest.raises(ValueError, match='Historical Padding'):
        CaffeRunner(unsupported)


def test_non_finite_prediction_rejected(exported, tmp_path):
    _ = exported
    deploy = tmp_path / 'nan.prototxt'
    deploy.write_text('input: "data" input_shape { dim:1 dim:1 dim:3 dim:4 }\n'
                      'layer { name: "root" type:"Power" bottom:"data" top:"output" '
                      'power_param { power:0.5 } }')
    with pytest.raises(FloatingPointError, match='non-finite output'):
        CaffeRunner(deploy).predict(-np.ones((1, 1, 3, 4), np.float32))


def test_cli_and_file_io(exported, tmp_path):
    fixtures, _ = exported
    directory = fixtures[3]['directory']
    source, destination = tmp_path / 'input.npy', tmp_path / 'prediction.npy'
    data = pattern_data('horizontal_ramp', 3, height=9, width=14)
    np.save(source, data)
    command = [sys.executable, str(ROOT / 'tools/run_caffe.py'),
               str(directory / 'runtime_deploy.prototxt'), str(source), str(destination),
               '--weights', str(directory / 'coefficients.caffemodel')]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)['finite'] is True
    np.testing.assert_allclose(np.load(destination), reference(data, fixtures[3]['manifest'])['product'],
                               rtol=4e-6, atol=3e-5)
    np.save(source, np.full((1, 3, 3, 4), np.nan))
    rejected = tmp_path / 'rejected.npy'
    command[4] = str(rejected)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 1 and 'non-finite' in completed.stderr
    assert not rejected.exists()


def test_cli_failed_save_preserves_previous_output(exported, tmp_path, monkeypatch):
    fixtures, _ = exported
    directory = fixtures[1]['directory']
    source, destination = tmp_path / 'input.npy', tmp_path / 'existing.npy'
    np.save(source, pattern_data('noise', 1))
    np.save(destination, np.array([42.0], dtype=np.float32))
    previous = destination.read_bytes()

    def failed_save(stream, *args, **kwargs):
        stream.write(b'partial write before disk error')
        raise OSError('injected write failure')

    monkeypatch.setattr(run_caffe.np, 'save', failed_save)
    result = run_caffe.main([str(directory / 'runtime_deploy.prototxt'), str(source),
                             str(destination), '--weights', str(directory / 'coefficients.caffemodel')])
    assert result == 1
    assert destination.read_bytes() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == ['existing.npy', 'input.npy']
