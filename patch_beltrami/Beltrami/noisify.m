% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

function z=noisify(y,sigma,sd)
randn('seed', sd);
z = y + (sigma)*randn(size(y));
z=max(min(z,255),0);