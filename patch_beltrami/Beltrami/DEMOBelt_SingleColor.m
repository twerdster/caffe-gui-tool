%% DEMO: Patch-Beltrami denoising (Color version)
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

clear all
close all
clc

%% Setup parameters (you can play with windowsize but I dont recommend it)
load paramsFor10ItersColor.mat

sigma = 20; % Set a sigma to inject noise. Or just use your own image and call it z

windowSize=3;
iterations=10;

bet=f2(sigma,1);%240
alpha=f2(sigma,2);


%% Input original
y = 255*double(im2double(imread('BM3D_images/image_Lena512rgb.png')));

%% Add noise
randn('seed', 0);
z = y + (sigma)*randn(size(y));
z=max(min(z,255),0);

%% Denoise with beltrami patch filter
tic;
y_est=beltColor(z,bet,alpha,iterations,windowSize);
dt=toc;

%% Get results

PSNR = 10*log10(255^2/mean((y(:)-y_est(:)).^2));
PN=  10*log10(255^2/mean((y(:)-z(:)).^2));

fprintf('Sigma:%2.0f\nNoise:%3.2fdb\nFiltered PSNR:%3.2fdb\nBeta:%f \nIters:%3.0f \nImgSize:(%i,%i)\nWindowSize:%i\nTime:%4.2f \n\n',sigma,PN,PSNR,bet,iterations,size(z,2),size(z,1),windowSize,dt);

%% Display output
imshow(uint8(abs(cat(2,z,y_est,y))));xlabel(num2str(PSNR));pause(0.00001);
title ('Noisy, Patch-Beltrami, Original');
