%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Setup   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cid = 0;

cid = cid+1;
% Test 256
config{cid}.path = 'data/68_IMGS/';
config{cid}.fname = 'data/DataTest256.h5';
config{cid}.sigma = 25;
config{cid}.patchSize = 256;
config{cid}.chans = 1;
config{cid}.onePatch = true;
config{cid}.shuffle = false;

cid = cid+1;
% Train
config{cid}.path = 'data/400_IMGS/';
config{cid}.fname = 'data/DataTrain.h5';
config{cid}.sigma = 25;
config{cid}.patchSize = 64;
config{cid}.chans = 1;
config{cid}.onePatch = false;
config{cid}.shuffle = true;

cid = cid+1;
% Test
config{cid}.path = 'data/68_IMGS/';
config{cid}.fname = 'data/DataTest.h5';
config{cid}.sigma = 25;
config{cid}.patchSize = 64;
config{cid}.chans = 1;
config{cid}.onePatch = false;
config{cid}.shuffle = true;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for cfg = 1:length(config)
    path = config{cfg}.path;
    fname = config{cfg}.fname;
    sigma = config{cfg}.sigma;
    patchSize = config{cfg}.patchSize;
    chans = config{cfg}.chans;
    onePatch = config{cfg}.onePatch;
    shuffle = config{cfg}.shuffle;
    
    fprintf('Creating data for config: \n');
    config{cfg}
    %% Get images
    clean = [];
    
    d = dir([path '*.png']);
    warning off;
    h = waitbar(0,'Getting image patches');
    for i=1:length(d)
        file = [path d(i).name];
        img = double(imread(file));
        patches = createPatches(img,[patchSize patchSize], true, onePatch);
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
    if shuffle
        randn('seed', 0);
        shuffle_idx = randperm(numPatches);
        clean = clean(:,:,:,shuffle_idx);
        noisy = noisy(:,:,:,shuffle_idx);
    end
    
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
end


