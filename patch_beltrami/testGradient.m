

%% Add paths
addpath(genpath('/home/gipadmin/forks/caffe/matlab'))


%% Reset caffe
caffe.set_mode_gpu()
caffe.reset_all()


%% Setup network
deployName = 'test.prototxt';
f=fopen(deployName,'w');
fprintf(f,[...
'name: "PBSolve_multi"\n' ...
'input: "data"\n' ...
'input_dim: 1\n' ...
'input_dim: 1\n' ...
'input_dim: 5\n' ...
'input_dim: 5\n' ...
'\n' ...
'force_backward: true\n' ...
'layer {\n' ...
'    name: "padding"\n' ...
'    type: "Padding"\n' ...
'    top: "padding"\n' ...
'    bottom: "data"\n' ...
'\n' ...
'    padding_param {\n' ...
'        pad_method: CONSTANT\n' ...
'        pad_l: 10\n' ...
'        pad_r: 2\n' ...
'        pad_t: 2\n' ...
'        pad_b: 10\n' ...
'        pad_value: -25.0\n' ...
'    }\n' ...
'}']);
fclose(f);

outLayer = 'padding';
inLayer = 'data';
EPS = 1e-2;
LOSS = @(Y) 0.5*sum(sum((Y.^2)));


net = caffe.Net(deployName,'test');

% Useful calls:
% net.layer_names
% a=net.layers('Ix').params.get_data();
% net.blob_names
% a=net.blobs('output_flow').params.get_data();

fprintf('\nStarting test\n\n');
fprintf('Input:\n');

rand('seed',0);
z = reshape(rand(1,25),[5 5])*100
net.blobs(inLayer).set_data(z);
net.forward_prefilled();

fprintf('Contents of output:\n');
O = net.blobs(outLayer).get_data();

fprintf('Setting output diff (as if 0 was required output)\n');
net.blobs(outLayer).set_diff(O);
net.blobs(outLayer).get_diff()

fprintf('Backward pass\n');
net.backward_prefilled();

fprintf('Updated diff on the input\n');
grad_analytic=net.blobs(inLayer).get_diff()


for i=1:length(z(:))
   zeps = z;
   zeps(i) = zeps(i)+EPS;
   net.blobs('data').set_data(zeps);
   net.forward_prefilled();
   Oforward(i) = LOSS(net.blobs(outLayer).get_data());
   
   zeps = z;
   zeps(i) = zeps(i)-EPS;
   net.blobs('data').set_data(zeps);
   net.forward_prefilled();
   Obackward(i) = LOSS(net.blobs(outLayer).get_data());
end
grad_numeric = reshape((Oforward-Obackward)/(2.0*EPS),[5 5])

fprintf('Percent error of numeric gradient over analytic (backward)\n');
int32(abs(grad_numeric-grad_analytic)./(grad_numeric+EPS/100)*100)

