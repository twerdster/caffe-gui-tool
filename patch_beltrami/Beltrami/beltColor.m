% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

function O=beltColor(A,beta,alpha,iterations,ws)
I=double(A); %work in regular domain
R=I(:,:,1);G=I(:,:,2);B=I(:,:,3);

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
    if 1%mod(p,2)==1
        Rx=Dmx(R);    Ry=Dmy(R);
        Gx=Dmx(G);    Gy=Dmy(G);
        Bx=Dmx(B);    By=Dmy(B);
        
        Ix2=paddedIntegralImage(Rx.^2+Gx.^2+Bx.^2,ws);
        IxIy=paddedIntegralImage(Rx.*Ry+Gx.*Gy+Bx.*By,ws);
        Iy2=paddedIntegralImage(Ry.^2+Gy.^2+By.^2,ws);
        
        g11=beta+Ix2(indA)+Ix2(indD)-Ix2(indB)-Ix2(indC);
        g12=IxIy(indA)+IxIy(indD)-IxIy(indB)-IxIy(indC);
        g22=beta+Iy2(indA)+Iy2(indD)-Iy2(indB)-Iy2(indC);
        
        gm05=(g11.*g22-g12.^2).^(-0.5);
        if ~all(isreal(gm05(:)))
            fprintf('Not Real: %i\n',p);
        end
    end
    R=belt(dt,R,Rx,Ry,gm05,g11,g12,g22);
    G=belt(dt,G,Gx,Gy,gm05,g11,g12,g22);
    B=belt(dt,B,Bx,By,gm05,g11,g12,g22);
    
end

O(:,:,1)=R;    O(:,:,2)=G;    O(:,:,3)=B;
