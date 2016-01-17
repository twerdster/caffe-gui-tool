% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

function O=beltBW(A,beta,alpha,iterations,ws)
I=double(A); %work in regular domain
%I=log2(double(A)+1); %work in log domain

Dmx=@(P) P-P([1 1:end-1],:);
Dmy= @(P) (Dmx(P'))';
Dpx= @(P) P([2:end end],:) -P;
Dpy= @(P) (Dpx(P'))';
belt=@(Dt,Im,Imx,Imy,Gm05,G11,G12,G22) ...
    Im+Dt*Gm05.*(Dpx(Gm05.*(G22.*Imx-G12.*Imy))+Dpy(Gm05.*(-G12.*Imx+G11.*Imy)));

paddedIntegralImage=@(I,ws) cumsum(cumsum(padarray(padarray(I,[ws ws]),[1 1],'pre')),2);

dt=alpha*beta;

[i,j]=meshgrid(1:size(I,1),1:size(I,2));
i=i';j=j';
sz=[size(I,1)+ws*2+1  size(I,2)+ws*2+1];

%integral image indices for calculating the sum around a pixel
indA=sub2ind(sz,i,j);
indB=sub2ind(sz,i,j+ws*2+1);
indC=sub2ind(sz,i+ws*2+1,j);
indD=sub2ind(sz,i+ws*2+1,j+ws*2+1);

for p=1:iterations
    
    Ix=Dmx(I);
    Iy=Dmy(I);
    Ix2=paddedIntegralImage(Ix.^2,ws);
    IxIy=paddedIntegralImage(Ix.*Iy,ws);
    Iy2=paddedIntegralImage(Iy.^2,ws);
    
    g11=beta+Ix2(indA)+Ix2(indD)-Ix2(indB)-Ix2(indC);
    g12=IxIy(indA)+IxIy(indD)-IxIy(indB)-IxIy(indC);
    g22=beta+Iy2(indA)+Iy2(indD)-Iy2(indB)-Iy2(indC);
    
    gm05=(g11.*g22-g12.^2).^(-0.5);
    
    I=belt(dt,I,Ix,Iy,gm05,g11,g12,g22);
   
    %I=max(min(I,8),0);
end

O=I;
%O(:,:,1)=(2.^I)-1;
