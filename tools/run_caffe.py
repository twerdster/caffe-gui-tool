"""Run an exported standard-layer Caffe deployment graph on the CPU.

Install numpy and opencv-python-headless>=4.10,<5. OpenCV 5 removed its
Caffe importer. This is inference only: supply trained or explicitly chosen
weights for parameterized layers. The research fork's Padding layer is not
implemented by this runner. Arrays use ordinary N,C,H,W order; no image
scaling, channel conversion, clipping, or historical wrapper is applied.
Grouped convolutions require each layer's input dimensions to be at least
its kernel dimensions: OpenCV's CPU depthwise path is unreliable below that.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np


def _tokens(source: str) -> list[str]:
    # Keep quoted strings intact so comments/parameter names inside a path
    # cannot be mistaken for actual layer declarations.
    return [token for token in re.findall(
        r'"(?:\\.|[^"\\])*"|#[^\n]*|[{}:]|[^\s{}:#]+', source
    ) if not token.startswith('#')]


class CaffeRunner:
    """Reusable OpenCV CPU net; create separate instances for concurrent use."""

    def __init__(self, deploy: str | Path, weights: str | Path | None = None):
        deploy = Path(deploy)
        if not deploy.is_file():
            raise ValueError(f'Deployment file does not exist: {deploy}')
        tokens = _tokens(deploy.read_text())
        layer_types = {
            tokens[index + 2].strip('"')
            for index, token in enumerate(tokens[:-2])
            if token == 'type' and tokens[index + 1] == ':'
        }
        if 'Padding' in layer_types:
            raise ValueError(
                'Historical Padding requires the research Caffe fork; '
                'this standard-layer OpenCV runner does not support it.'
            )
        if layer_types & {'Data', 'ImageData', 'HDF5Data', 'HDF5Output'}:
            raise ValueError('Use a deployment graph with explicit input blobs, not a training graph.')
        parameterized = layer_types & {'Convolution', 'Deconvolution', 'InnerProduct',
                                       'PReLU', 'BatchNorm', 'Scale', 'Embed'}
        if weights is None and parameterized:
            raise ValueError(
                'Supply an explicit .caffemodel with --weights for parameterized '
                'layers. OpenCV does not load inline prototxt blobs or initialize fillers.'
            )
        if weights is not None and not Path(weights).is_file():
            raise ValueError(f'Weights file does not exist: {weights}')
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError('Install opencv-python-headless>=4.10,<5 to run Caffe graphs.') from error
        if not hasattr(cv2.dnn, 'readNetFromCaffe'):
            raise RuntimeError('OpenCV 5 removed Caffe import; install opencv-python-headless>=4.10,<5.')
        self.cv2 = cv2
        try:
            self.net = cv2.dnn.readNetFromCaffe(str(deploy), '' if weights is None else str(weights))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            # Fusion can overwrite a named convolution's buffer with its
            # activation after previous forward calls. Preserve layer outputs
            # when callers alternate between final and intermediate results.
            self.net.enableFusion(False)
            self._convolutions = {}
            for name in self.net.getLayerNames():
                identifier = self.net.getLayerId(name)
                layer = self.net.getLayer(identifier)
                if layer.type == 'Convolution' and layer.blobs:
                    shape = layer.blobs[0].shape
                    if len(shape) == 4:
                        self._convolutions[identifier] = (name, shape)
        except cv2.error as error:
            raise ValueError(f'Cannot import this Caffe graph/weights: {error}') from error

    def predict(self, data: np.ndarray, *, input_name: str = 'data',
                output_layer: str | None = None) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim != 4 or any(size <= 0 for size in data.shape):
            raise ValueError('Input must be a nonempty N,C,H,W array.')
        if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise ValueError('Input must contain real numeric values.')
        if not np.isfinite(data).all():
            raise ValueError('Input contains non-finite values.')
        with np.errstate(over='ignore', invalid='ignore'):
            data = np.ascontiguousarray(data, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError('Input is outside the finite float32 range.')
        try:
            if self._convolutions:
                identifiers, input_shapes, _ = self.net.getLayersShapes(tuple(data.shape))
                for identifier, shapes in zip(identifiers, input_shapes):
                    convolution = self._convolutions.get(int(identifier))
                    if convolution is None or not shapes:
                        continue
                    name, weights_shape = convolution
                    shape = shapes[0]
                    # Input C divided by the weight's C gives Caffe's group.
                    # Check the actual input to every convolution, including
                    # layers downstream of pooling/strides, before execution.
                    if len(shape) == 4 and shape[1] > weights_shape[1] and (
                        shape[2] < weights_shape[2] or shape[3] < weights_shape[3]
                    ):
                        raise ValueError(
                            f'OpenCV CPU grouped convolution {name!r} requires '
                            f'spatial dimensions of at least {weights_shape[2]} by '
                            f'{weights_shape[3]}; its input is {shape[2]} by {shape[3]}.'
                        )
            self.net.setInput(data, input_name)
            result = self.net.forward(output_layer) if output_layer else self.net.forward()
        except self.cv2.error as error:
            raise ValueError(f'Caffe inference failed; check shapes, layer support and weights: {error}') from error
        if not np.isfinite(result).all():
            raise FloatingPointError('Caffe inference produced non-finite output.')
        return result.copy()


def _save_atomic(path: Path, result: np.ndarray) -> None:
    """Keep an existing prediction intact if serialization or replacement fails."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                                         prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, result, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('deploy', type=Path)
    parser.add_argument('input', type=Path, help='Numeric .npy array in N,C,H,W order')
    parser.add_argument('output', type=Path, help='Destination .npy prediction')
    parser.add_argument('--weights', type=Path, help='Explicit .caffemodel; required for parameterized layers')
    parser.add_argument('--input-name', default='data')
    parser.add_argument('--output-layer', help='OpenCV layer name; defaults to the graph output')
    args = parser.parse_args(argv)
    try:
        if args.output.suffix != '.npy':
            raise ValueError('Output filename must end in .npy.')
        result = CaffeRunner(args.deploy, args.weights).predict(
            np.load(args.input, allow_pickle=False), input_name=args.input_name,
            output_layer=args.output_layer,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _save_atomic(args.output, result)
    except (ValueError, RuntimeError, FloatingPointError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    print(json.dumps({'shape': list(result.shape), 'dtype': str(result.dtype),
                      'finite': True, 'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
