%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Setup   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Train
config{1}.path = '400_IMGS/';
config{1}.fname = 'DataTrain.h5';
config{1}.sigma = 25;
config{1}.patchSize = 64;
config{1}.chans = 1;
config{1}.onePatch = false;
config{1}.shuffle = true;

% Test
config{2}.path = '68_IMGS/';
config{2}.fname = 'DataTest.h5';
config{2}.sigma = 25;
config{2}.patchSize = 64;
config{2}.chans = 1;
config{2}.onePatch = false;
config{2}.shuffle = true;

% Test 256
config{3}.path = '68_IMGS/';
config{3}.fname = 'DataTest256.h5';
config{3}.sigma = 25;
config{3}.patchSize = 256;
config{3}.chans = 1;
config{3}.onePatch = true;
config{3}.shuffle = false;

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








