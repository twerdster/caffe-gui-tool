% Aaron Wetzler, Ron Kimmel, SSVM 2011
% This code is for academic use only.
% Contact: aaronwetzler@gmail.com

function O=belt rami(A,beta,alpha,iterations)
O=log2(double(A)+1); %work in log domain
%O=double(A);
R=double(O(:,:,1));
G=double(O(:,:,2));
B=double(O(:,:,3));
dt=0.21*beta;
for p=1:iterations,
    Rx=Dmx(R);     Gx=Dmx(G);    Bx= Dmx(B);   
    Ry=Dmy(R);     Gy=Dmy(G);     By=Dmy(B);
    
    g11=beta+Rx.^2+Gx.^2+Bx.^2;
    g12=Rx.*Ry+Gx.*Gy+Bx.*By;
    g22=beta+Ry.^2+Gy.^2+By.^2;
    gm05=(g11.*g22-g12.^2).^(-0.5); %gm05=1/sqrt(g)
    
    R=belt(dt,R,Rx,Ry,gm05,g11,g12,g22);
    G=belt(dt,G,Gx,Gy,gm05,g11,g12,g22);
    B=belt(dt,B,Bx,By,gm05,g11,g12,g22);
end

O(:,:,1)=(2.^R)-1;O(:,:,2)=(2.^G)-1;O(:,:,3)=(2.^B)-1;

%-------------------------------------------------------------
%res=R+dt*beltramigR
function res=belt(dt,R,Rx,Ry,gm05,g11,g12,g22)
%This computes the numerical flow value for a given dimension R 
%using the beltrami operator
res=gm05.*(Dpx(gm05.*(g22.*Rx-g12.*Ry))+Dpy(gm05.*(-g12.*Rx+g11.*Ry)));

%Now we compute the actual change based on the time step
res=R+dt*res;

%This restricts the values to be between 1 and 2^8 in the final image
%because we are working in logspace. If we want to work in regular space
%we should change the maximum to 255
res=max(min(res,8),0);

%-------------------------------------------------------------
function f=Dmx(P)
f=P-P([1 1:end-1],:);

%-------------------------------------------------------------
function f=Dmy(P)
f=(Dmx(P'))';

%-------------------------------------------------------------
function f=Dpx(P)
f=P([2:end end],:) -P;

%-------------------------------------------------------------
function f=Dpy(P)
f=(Dpx(P'))';
