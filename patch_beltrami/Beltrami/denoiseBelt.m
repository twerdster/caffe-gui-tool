% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

function [I_est,PSNR,PN,dt]=denoiseBelt(I,sigma)

sigma=ceil(sigma);
if sigma<1 error('Sigma must be larger than or equal to 1')
elseif sigma>100 error('Sigma must be less than or equal to 100');
end

len=length(size(I));
if (len~=2 && len~=3) error('Image must be either graylevel or color.'); end

windowSize=3;
iterations=10;
randn('seed', 0);
z = I + (sigma)*randn(size(I));
z=max(min(z,255),0);

if len==2
    load paramsFor10ItersBW.mat;
    bet=f(sigma,1);
    alpha=f(sigma,2);
    tic; y_est=beltBW(z,bet,alpha,iterations,windowSize);dt=toc;
else
    load paramsFor10ItersColor.mat;
    bet=f2(sigma,1);
    alpha=f2(sigma,2);
    tic; y_est=beltColor(z,bet,alpha,iterations,windowSize); dt=toc;
end

PSNR = 10*log10(255^2/mean((I(:)-y_est(:)).^2));
PN=  10*log10(255^2/mean((I(:)-z(:)).^2));
I_est=y_est;
