
%% Add paths
addpath(genpath('/home/gipadmin/forks/caffe/matlab'))


%% Reset caffe
caffe.set_mode_gpu()
caffe.reset_all()


%% Setup network
deployName = 'quick_solve_deploy.prototxt';
modelName = 'aaron__iter_1.caffemodel';
%modelName = 'rom__iter_1.caffemodel';
trainORtest = 'test';

setNameAndInputDims(deployName,'quick_solve',[1,3,512,512]);

% The _tmp is the temp file with the dimensions added. Theres probably
% a nicer way of doing this.
net = caffe.Net(...
    [deployName '_tmp'],...
    modelName,trainORtest);


%% Run data through network

I = h5read('PatchBeltrami.h5','/data');
I = I(:,:,:,1);

n=100;
tic;
im = I(:,:,:,1);

for i=1:n

    net.blobs('data').set_data(im);
    net.forward_prefilled();
    
    delta = net.blobs('output_flow').get_data();
    
    im = im + delta;
    imshow(im/255);
    fprintf('Iteration:%i\n',i);
    drawnow
end
t2=toc;

fprintf('Total time: %f\n',(t2)/n);

%% Look at other elements of the data

% net.blob_names

% net.layer_names

