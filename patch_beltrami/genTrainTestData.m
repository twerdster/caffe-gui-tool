%we are creating patches here
%as well as original image
%each pair consists of the clean and corrupted patch
%each dataset has its sigma, patch size,
%can also do shotnoise

%Initialize data params
splitPoint = 4; % Before this the images are train and after which the images are used for test
names= [ ...
    {'image_Baboon512rgb'} ...
    {'image_F16_512rgb'} ...
    {'image_House256rgb'} ...
    {'image_Lena512rgb'} ...
    {'image_Peppers512rgb'} ...
    ];
patchSize = 64;
sigma = 20;
numPatches = 5*(512^2)/(patchSize^2);
patchesD_train = zeros(patchSize, patchSize, 3, numPatches, 'single');
patchesL_train = zeros(patchSize, patchSize, 3, numPatches, 'single');


% Setup files
delete DataTrain.h5
delete DataTest.h5
h5create('DataTrain.h5', '/data',  [patchSize,patchSize,3,inf], 'Chunksize',[patchSize patchSize 3 1], 'Datatype', 'single');
h5create('DataTrain.h5', '/label', [patchSize,patchSize,3,inf], 'Chunksize',[patchSize patchSize 3 1], 'Datatype', 'single');
h5create('DataTest.h5',  '/data',  [patchSize,patchSize,3,inf], 'Chunksize',[patchSize patchSize 3 1], 'Datatype', 'single');
h5create('DataTest.h5',  '/label', [patchSize,patchSize,3,inf], 'Chunksize',[patchSize patchSize 3 1], 'Datatype', 'single');

% Start data creation
randn('seed', 0);

cntTrain = 0; % train patch number
cntTest = 0; %  test patch number
cnt_ = 0;
for i=1:length(names)
    % get file name and image
    fprintf('Processing: %s\n',['BM3D_images/' names{i} '.png']);
    eval(['y = 255*double(im2double(imread(''BM3D_images/' names{i} '.png'')));']);
    
    % Add noise
    z = y + (sigma)*randn(size(y));
    z=max(min(z,255),0);
    bet=f2(sigma,1);%ignore 
    alpha=f2(sigma,2);%ignore
    
    % Steps through images and extracts patches non overlapping
    for u=1:patchSize:size(y,1)
        for v = 1:patchSize:size(y,2)
            fprintf('%i %i %i\n',i,u,v);
            d  = single(z(u:u+patchSize-1,  v:v+patchSize-1, :));
            gt = single(y(u:u+patchSize-1,  v:v+patchSize-1, :));
            
            cnt_ = cnt_ + 1;
            patchesD(:,:,:,cnt_) = d;
            patchesL(:,:,:,cnt_) = gt;
            
            % Send to train or test. We dont want same images to be used
            % for both
            if i<5 
                fname = 'DataTrain.h5';
                cntTrain = cntTrain + 1;
                cnt = cntTrain;
            else
                fname = 'DataTest.h5';
                cntTest = cntTest + 1;
                cnt = cntTest;
            end
            
            h5write(fname, '/data',   d ,[1 1 1 cnt],[patchSize patchSize 3 1]);
            h5write(fname, '/label', gt ,[1 1 1 cnt],[patchSize patchSize 3 1]);
                        
        end
    end
end

fprintf('train: %i, test: %i\n',cntTrain, cntTest);







