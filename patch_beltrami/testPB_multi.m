

%% Add paths
addpath(genpath('/home/gipadmin/forks/caffe/matlab'))


%% Reset caffe
caffe.set_mode_gpu()
caffe.reset_all()


%% Setup network
%deployName = 'quick_solve_deploy.prototxt';
%modelName = 'aaron__iter_1.caffemodel';

deployName = 'PBSolve_multi/PBSolve_multi_deploy.prototxt';
%modelName = 'PBSolve_multi/aaron__iter_13.caffemodel';
modelName = 'PBSolve_multi/aaron__iter_1000.caffemodel';

trainORtest = 'test';

setNameAndInputDims(deployName,'PBSolve_multi',[1,1,256,256]);

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

I_data = h5read('DataTest256.h5','/data');
I_label = h5read('DataTest256.h5','/label');

sigma = 25;

warning off;
for i=1:size(I_data,4)
    y = I_label(:,:,:,i);
    z = I_data(:,:,:,i);
    
    net.blobs('data').set_data(z);
    net.forward_prefilled();
    
    
    % delta = net.blobs('output_flow00').get_data();
    % y_est = y_est + delta*d.dt;
%         for j=1:10
%             y_est = net.blobs('finalOutput').get_data();
%             net.blobs('data').set_data(y_est);
%             net.forward_prefilled();
%         end
%     
%     B.Ix=net.blobs('Ix').get_data();
%     B.Iy=net.blobs('Iy').get_data();
%     B.IxIy=net.blobs('IxIy').get_data();
%     B.Ix2=net.blobs('Ix2').get_data();
%     B.Iy2=net.blobs('Iy2').get_data();
%     B.g11=net.blobs('g11').get_data();
%     B.g12=net.blobs('g12').get_data();
%     B.g22=net.blobs('g22').get_data();
%     B.gm05=net.blobs('gm05').get_data();
%     B.I=net.blobs('finalOutput').get_data();
%     
    y_est = net.blobs('finalOutput000000000').get_data();
    
    PSNR = 10*log10(255^2/mean((y(:)-y_est(:)).^2));
    PN=  10*log10(255^2/mean((y(:)-z(:)).^2));
    psnr(i) = PSNR;
    pn(i) = PN;
    fprintf('#:%i Sigma:%2.0f\nNoise:%3.2fdb\nFiltered PSNR:%3.2fdb\n',i,sigma,PN,PSNR);
    imshow(uint8(abs(y_est)));xlabel(num2str(PSNR));
    sssss
    drawnow
end
warning on
t2=toc;

fprintf('Total time: %f\n',(t2)/n);

%% Look at other elements of the data

% net.blob_names

% net.layer_names

