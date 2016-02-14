I(:,:,:,1) = imread('image_Lena512rgb.png');
I(:,:,:,2) = imread('image_Peppers512rgb.png');
I(:,:,:,3) = imread('image_F16_512rgb.png');
I(:,:,:,4) = imread('image_Baboon512rgb.png');

I = I;
I = single(I);


h5create('Data.h5', '/data', [512,512,3,inf], 'Chunksize',[512 512 3 1],'Datatype','single');
h5create('Data.h5', '/label', [512,512,3,inf], 'Chunksize',[512 512 3 1],'Datatype','single');

for i=1:4
    h5write('Data.h5', '/data', I(:,:,:,i) ,[1 1 1 i],[512 512 3 1]);
    h5write('Data.h5', '/label', I(:,:,:,i) ,[1 1 1 i],[512 512 3 1]);
end