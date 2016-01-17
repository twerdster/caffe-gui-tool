% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

function O=beltColorConv2(A,beta,alpha,iterations,ws)
I=double(A); %work in regular domain
R=I(:,:,1);G=I(:,:,2);B=I(:,:,3);

Dmx=@(P) P-P([1 1:end-1],:);
Dmy= @(P) (Dmx(P'))';
Dpx= @(P) P([2:end end],:) -P;
Dpy= @(P) (Dpx(P'))';
belt=@(Dt,Im,Imx,Imy,Gm05,G11,G12,G22) ...
    Im+Dt*Gm05.*(Dpx(Gm05.*(G22.*Imx-G12.*Imy))+Dpy(Gm05.*(-G12.*Imx+G11.*Imy)));

dt=alpha*beta;
h=ones(ws*2+1);

for p=1:iterations
    
    Rx=Dmx(R);  Ry=Dmy(R);
    Gx=Dmx(G);  Gy=Dmy(G);
    Bx=Dmx(B);   By=Dmy(B);
    
    g11=beta+conv2(Rx.^2+Gx.^2+Bx.^2,h,'same');
    g12=conv2(Rx.*Ry+Gx.*Gy+Bx.*By,h,'same');
    g22=beta+conv2(Ry.^2+Gy.^2+By.^2,h,'same');
    
    gm05=(g11.*g22-g12.^2).^(-0.5);
    
    R=belt(dt,R,Rx,Ry,gm05,g11,g12,g22);
    G=belt(dt,G,Gx,Gy,gm05,g11,g12,g22);
    B=belt(dt,B,Bx,By,gm05,g11,g12,g22);
    
end

O(:,:,1)=R;O(:,:,2)=G;O(:,:,3)=B;

