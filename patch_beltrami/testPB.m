

%% Add paths
addpath(genpath('/home/gipadmin/forks/caffe/matlab'))


%% Reset caffe
caffe.set_mode_cpu()
caffe.reset_all()


%% Setup network
%deployName = 'quick_solve_deploy.prototxt';
%modelName = 'aaron__iter_1.caffemodel';

deployName = 'PBSolve/PBSolve_deploy.prototxt';
modelName = 'PBSolve/aaron__iter_26.caffemodel';

trainORtest = 'test';

setNameAndInputDims(deployName,'PBSolve',[1,3,512,512]);

% The _tmp is the temp file with the dimensions added. Theres probably
% a nicer way of doing this.
net = caffe.Net(...
    [deployName '_tmp'],...
    modelName,trainORtest);


%% Run data through network


% Useful calls:
% net.layer_names
% a=net.layers('Ix').params.get_data();
% net.blob_names
% a=net.blobs('output_flow').params.get_data();

I = h5read('PatchBeltrami.h5','/data');
I = I(:,:,:,1);

n=100;
sigma = 20;

tic;
y_est = I(:,:,:,1);

d = load('dataB');
y_est = d.z;
fprintf('Starting: Sigma:%2.0f ImgSize:(%i,%i)\n',sigma,size(d.z,2),size(d.z,1));

for i=1:n
    
    net.blobs('data').set_data(y_est);
    net.forward_prefilled();
    
    delta = net.blobs('output_flow').get_data();
    
    y_est = y_est + delta*d.dt;
    warning off;imshow(y_est/255);warning on
    
    
    PSNR = 10*log10(255^2/mean((d.y(:)-y_est(:)).^2));
    PN=  10*log10(255^2/mean((d.y(:)-d.z(:)).^2));
    fprintf('#:%i PSNR:%3.2fdb\n',i,PSNR);
    
    drawnow
end
t2=toc;

fprintf('Total time: %f\n',(t2)/n);

%% Look at other elements of the data

% net.blob_names

% net.layer_names

