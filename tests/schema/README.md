# Caffe export schema

`caffe.proto` is copied without modification from the author's historical
[Caffe fork](https://github.com/twerdster/caffe/blob/a6083b3ab99a85fc3bf55a9ff84251861d7c6b63/src/caffe/proto/caffe.proto),
revision `a6083b3ab99a85fc3bf55a9ff84251861d7c6b63`.
Its BSD license is retained in `LICENSES/Caffe-BSD.txt`.

This schema includes the research `Padding` layer. Parsing an export against
this schema validates its syntax and field types, not layer implementation,
tensor shapes, training convergence or compatibility with stock Caffe.
