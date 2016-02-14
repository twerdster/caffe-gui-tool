% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

function E = funcBeltBW(z,y,params,iters,ws)
bet=exp(params(1));%240
alpha=0.21;%exp(params(2));
tic;
y_est=beltBWConv2(z,bet,alpha,iters,ws);
dt=toc;
% Compute the putput PSNR
PSNR = 10*log10(255^2/mean((y(:)-y_est(:)).^2));
E=-PSNR;
fprintf('%3.3f -> %f , %f   Time:%2.2f\n',E,bet,alpha,dt);
