% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

% Please cite the following if you use it in your work.
% @inproceedings{Wetzler11,
%   author    = {A. Wetzler and
%                R. Kimmel},
%   title     = {Efficient Beltrami Flow in Patch-Space},
%   booktitle = {SSVM},
%   year      = {2011},
%   pages     = {134-143},
% }

%clear all
%close all
%clc
load paramsFor10ItersBW.mat

F=[];

I_data = h5read('DataTest256.h5','/data');
I_label = h5read('DataTest256.h5','/label');

for i=1:1:68
    y = I_label(:,:,:,i);
    
    idx = 1;
    for sigma=[ 25] %1:100
        
        z = I_data(:,:,:,i);
        %z=max(min(z,255),0);
        
        windowSize=3;
        iterations=10;
        bet=f(sigma,1);%240
        alpha=f(sigma,2);
        tic;
        y_est=beltBW(z,bet,alpha,iterations,windowSize);
      %   y_est=beltBWConv2(z,bet,alpha,iterations,windowSize);
        dt=toc;
        
        PSNR = 10*log10(255^2/mean((y(:)-y_est(:)).^2));
        PN=  10*log10(255^2/mean((y(:)-z(:)).^2));
        F(sigma,i)=PSNR;
      %  fprintf('Sigma:%i  PSNR:%f \n',sigma,PSNR);
        
        psnr(idx,i) = PSNR;
        pn(idx,i) = PN;
        idx = idx+1;
    end
    %fprintf('Sigma:%2.0f\nNoise:%3.2fdb\nFiltered PSNR:%3.2fdb\nBeta:%f \nIters:%3.0f \nImgSize:(%i,%i)\nWindowSize:%i\nTime:%4.2f \n\n',sigma,PN,PSNR,bet,iterations,size(z,2),size(z,1),windowSize,dt);
    
       
    fprintf('#:%i Sigma:%2.0f\nNoise:%3.2fdb\nFiltered PSNR:%fdb\n',i,sigma,PN,PSNR);
    imshow(uint8(abs(y_est)));xlabel(num2str(PSNR));pause(0.00001);
end
