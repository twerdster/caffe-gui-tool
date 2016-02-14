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

% Read a grayscale image and scale its intensities in range [0,1]
%y = 255*double(im2double(imread('BM3D_images\Cameraman256.png')));
clear all
close all
clc
load paramsFor10ItersColor.mat
names= [ ...
    {'image_Baboon512rgb'} ...
    {'image_F16_512rgb'} ...
    {'image_House256rgb'} ...
    {'image_Lena512rgb'} ...
    {'image_Peppers512rgb'} ...
    ];
F2=[];

for i=1:length(names)
    eval(['y = 255*double(im2double(imread(''BM3D_images\' names{i} '.png'')));']);
    
    for sigma=1:100
        randn('seed', 0);
   
        z = y + (sigma)*randn(size(y));
        z=max(min(z,255),0);
        
        windowSize=3;
        iterations=10;
        bet=f2(sigma,1);%240
        alpha=f2(sigma,2);
        tic;
        y_est=beltColor(z,bet,alpha,iterations,windowSize);
        dt=toc;
        
        PSNR = 10*log10(255^2/mean((y(:)-y_est(:)).^2));
        PN=  10*log10(255^2/mean((y(:)-z(:)).^2));
        F2(sigma,i)=PSNR;
         fprintf('Sigma:%i  PSNR:%f \n',sigma,PSNR);
    end
    fprintf('Sigma:%2.0f\nNoise:%3.2fdb\nFiltered PSNR:%3.2fdb\nBeta:%f \nIters:%3.0f \nImgSize:(%i,%i)\nWindowSize:%i\nTime:%4.2f \n\n',sigma,PN,PSNR,bet,iterations,size(z,2),size(z,1),windowSize,dt);
    
    % show the noisy image 'z' and the denoised 'y_est'
    %figure; imshow(uint8(z));
    imshow(uint8(abs(y_est)));xlabel(num2str(PSNR));pause(0.00001);
    save resultsFor10ItersAllImagesColor.mat F2
end

