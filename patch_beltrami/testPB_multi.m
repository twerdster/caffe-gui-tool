
%% Add paths
addpath(genpath('/home/gipadmin/forks/caffe/matlab'))

%% Reset caffe
caffe.set_mode_cpu()
caffe.reset_all()

%% Setup network
deployName = 'PBSolve_multi/PBSolve_multi_deploy.prototxt';
modelName = 'PBSolve_multi/aaron__iter_10000.caffemodel';

trainORtest = 'test';

setNameAndInputDims(deployName,'PBSolve_multi',[1,1,256,256]);

% The _tmp is the temp file with the dimensions added. Theres probably
% a nicer way of doing this.
net = caffe.Net(...
    [deployName '_tmp'],...
    modelName,trainORtest);

getData = @(net,name,ind) net.layer_vec(net.name2layer_index(name)).params(ind).get_data();
getBlob = @(net,name) net.blob_vec(net.name2blob_index(name)).get_data();

%% Run data through network and get PSNR

I_data = h5read('data/DataTest256.h5','/data');
I_label = h5read('data/DataTest256.h5','/label');

sigma = 25;

figure(1)
warning off;
n=68;

load('OtherMethods'); % Loads A -> NLD, B -> BM3D, C->PatchBeltrami

tic;
for i=1:n
    y = I_label(:,:,:,(i));
    z = I_data(:,:,:,(i));
    
    net.blobs('data').set_data(z);
    net.forward_prefilled();
    y_est = net.blobs('finalOutput0000').get_data();
    
%     delta = net.blobs('output_flow00').get_data();
%     y_est = y_est + delta*d.dt;
%     for j=1:0 %2
%         y_est = net.blobs('finalOutput').get_data();
%         net.blobs('data').set_data(y_est);
%         net.forward_prefilled();
%     end
    % Get debug info from inside the network
    %     res.Ix=net.blobs('Ix').get_data();
    %     res.Iy=net.blobs('Iy').get_data();
    %     res.IxIy=net.blobs('IxIy').get_data();
    %     res.Ix2=net.blobs('Ix2').get_data();
    %     res.Iy2=net.blobs('Iy2').get_data();
    %     res.g11=net.blobs('g11').get_data();
    %     res.g12=net.blobs('g12').get_data();
    %     res.g22=net.blobs('g22').get_data();
    %     res.gm05=net.blobs('gm05000').get_data();
    %     res.I=net.blobs('finalOutput').get_data();
    %
    
    PSNR = 10*log10(255^2/mean((y(:)-y_est(:)).^2));
    PN=  10*log10(255^2/mean((y(:)-z(:)).^2));
    psnr(i) = PSNR;
    pn(i) = PN;
    
    fprintf('#:%i RPB:%fdb  TRD:%fdb  BM3D:%f  PB:%f\n',i,psnr(i),A.A{5}.psnr((i)),B.B.psnr((i)),C.psnr((i)));
    
    imshow(uint8(abs(y_est)));xlabel(num2str(PSNR));
    
    drawnow
    I_{i} = y_est;
end
warning on
t2=toc;

fprintf('Average time: %f\n',(t2)/n);



%% Compare to Non-Linear Diffusion
RPB_psnr = psnr;
TRF_psnr = A.A{5}.psnr;
BM3D_psnr = B.B.psnr;
PB_psnr = C.psnr;

[a,ind] = sort(TRF_psnr-RPB_psnr);
b = BM3D_psnr - RPB_psnr; b=b(ind);
c = PB_psnr - RPB_psnr; c=c(ind);

fprintf('Median: %f\n',median(a));
figure(2);

hold on;
plot(a,'b'); %TRF
plot(b,'r'); %BM3d
plot(c,'g'); % PB
plot(b*0,'c');
fprintf('Beating BM3D: %i\n',sum(RPB_psnr > BM3D_psnr ) );
fprintf('Beating TRD: %i\n', sum(RPB_psnr > TRF_psnr  ) );
fprintf('Beating PB: %i\n', sum(RPB_psnr > PB_psnr  ) );

%A{2} = load('PSNR_results__JointTraining_5x5_400_180x180_stage=5_sigma=25.mat');
%A{3} = load('PSNR_results__JointTraining_5x5_400_180x180_stage=8_sigma=25.mat');
%A{4} = load('PSNR_results__JointTraining_7x7_400_180x180_stage=5_sigma=25.mat');
%A{5} = load('PSNR_results__JointTraining_7x7_400_180x180_stage=5_sigma=25.mat');

%% Look at other elements of the data

% net.blob_names

% net.layer_names


I_data = h5read('data/DataTest256.h5','/data');
I_label = h5read('data/DataTest256.h5','/label');

i = 3;
y = I_label(:,:,:,i);
z = I_data(:,:,:,i);

yr = rpb.I_{i};%yr(yr>255)=255;yr(yr<0)=0;
yt = trd.I_{i};%yt(yt>255)=255;yt(yt<0)=0;
yb = 255*bm3d.I_{i};%yt(yt>255)=255;yt(yt<0)=0;

delrpb = abs(yr-I_label(:,:,:,i));
deltrd = abs(yt-I_label(:,:,:,i));
delbm3d = abs(yb-I_label(:,:,:,i));
PSNRr = 10*log10(255^2/mean((y(:)-yr(:)).^2))
PSNRt = 10*log10(255^2/mean((y(:)-yt(:)).^2))
PSNRb = 10*log10(255^2/mean((y(:)-yb(:)).^2))
PSNRr-PSNRt
PSNRr-PSNRb

yy=[50  201];
xx=[1   82];
out = z(xx(1):xx(2),yy(1):yy(2));
out = cat(2,out,yr(xx(1):xx(2),yy(1):yy(2)));
out = cat(2,out,yb(xx(1):xx(2),yy(1):yy(2)));
out = cat(2,out,yt(xx(1):xx(2),yy(1):yy(2)));
out = cat(2,out,y(xx(1):xx(2),yy(1):yy(2)));
imshow([out],[0 255]);

