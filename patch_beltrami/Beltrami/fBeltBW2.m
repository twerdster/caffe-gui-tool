% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

function E = fBeltBW2(In,I,params,iters,ws)
dt=exp(params(1));
alpha=exp(params(2));
tic;
I_est=bBW(In,alpha,dt,iters,ws);
t=toc;
E = -10*log10(255^2/mean((I(:)-I_est(:)).^2));
fprintf('%3.3f -> %f , %f   Time:%2.2f\n',-E,dt,alpha,t);
