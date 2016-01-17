% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

function O=bBW(A,alpha,dt,iterations,ws)
I=double(A); %work in regular domain
%I=log2(I+1);
Dmx=@(P) P-P([1 1:end-1],:);
Dmy= @(P) (Dmx(P'))';
Dpx= @(P) P([2:end end],:) -P;
Dpy= @(P) (Dpx(P'))';
belt=@(Dt,Im,Imx,Imy,Gm05,G11,G12,G22) ...
    Im+Dt*Gm05.*(Dpx(Gm05.*(G22.*Imx-G12.*Imy))+Dpy(Gm05.*(-G12.*Imx+G11.*Imy)));

h=ones(ws*2+1);
for p=1:iterations
    
    Ix=Dmx(I);
    Iy=Dmy(I);
   
    g11=1+alpha^2*conv2(Ix.^2,h,'same');
    g12=alpha^2*conv2(Ix.*Iy,h,'same');
    g22=1+alpha^2*conv2(Iy.^2,h,'same');
    
    gm05=(g11.*g22-g12.^2).^(-0.5);
    
    I=belt(dt,I,Ix,Iy,gm05,g11,g12,g22);
end
O=I;
%O=2.^I-1;
