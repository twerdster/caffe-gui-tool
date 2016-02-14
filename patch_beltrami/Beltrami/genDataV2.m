%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Setup   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path = '400_IMGS/'; fname = 'DataTrain.h5'; 
%path = '68_IMGS/';  fname = 'DataTest.h5';

sigma = 25;
patchSize = 64;
chans = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Get training images
clean = [];

d = dir([path '*.png']);
warning off;
h = waitbar(0,'Getting image patches');
for i=1:length(d)
    file = [path d(i).name];
    img = double(imread(file));
    patches = createPatches(img,[patchSize patchSize], true);
    clean = cat(4,clean,patches);
    imshow(img/255);
    waitbar(i/length(d));
    drawnow
end
close(h)
warning on;
clean = single(clean);
numPatches = size(clean,4);

%% Noisify
randn('seed', 0);
noisy = clean + sigma*randn(size(clean));
% noisy = double(uint8(noisy));

%% Shuffle data
randn('seed', 0);
shuffle_idx = randperm(numPatches);
clean = clean(:,:,:,shuffle_idx);
noisy = noisy(:,:,:,shuffle_idx);

%% Setup files
delete(fname)
h5create(fname, '/data',  [patchSize,patchSize,chans,inf], 'Chunksize',[patchSize patchSize chans 1], 'Datatype', 'single');
h5create(fname, '/label', [patchSize,patchSize,chans,inf], 'Chunksize',[patchSize patchSize chans 1], 'Datatype', 'single');

h = waitbar(0,'Writing HDF5 files');
for i=1:numPatches
    h5write(fname, '/data',  noisy(:,:,:,i) ,[1 1 1 i],[patchSize patchSize chans 1]);
    h5write(fname, '/label', clean(:,:,:,i) ,[1 1 1 i],[patchSize patchSize chans 1]);
    waitbar(i/numPatches);
end
close(h);








