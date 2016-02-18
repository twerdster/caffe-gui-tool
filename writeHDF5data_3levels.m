
close all

%expName = 'FinalData_NYU_test/';
%expName = 'FinalData_NYU_derotated_test/';
%expName = 'FinalData_NYU_train/';
expName = 'FinalData_NYU_derotated_train/';

outExpName = 'HDF5_HM/';
dataDir = ['/home/gipuser/Documents/Original/' expName];
outDir = ['/home/gipuser/Documents/' outExpName expName];
mkdir(outDir);

d=dir([dataDir 'Data*.mat']);

len = length(d);
tform = maketform('affine',eye(3));

EXAMPLE_LIM = 20000; % VERY NB: Allow only 20K examples per HDF5 file
shuffle = true;


imCnt = 0;
oldId = -1;
newId = -1;
centerInd = 6;
maxDepth = 250/2; % smallest to largest would be 250. 
hsz = 18;
imsz = 96;
hmpInds = [1 2 3 4 5];
lenh = length(hmpInds);

h = fspecial('gaussian',9,5);

%error('This file needs to be run once on FinalDataHDF5_HM and then once on FinalDataHDF5_HM_derotated');
%%

rand('seed',0); % This will give the same random ordering everytime for the same dataset
if shuffle
    permInd = int32(randperm(len));
else
    permInd=1:len;
end

[~,revInd]=sort(permInd);

for cnt=permInd
    fprintf('%i/%i\n',imCnt,len);
    load([dataDir d(cnt).name]);
    hmap = hmap(:,:,hmpInds);
    msk = depth==0 | ~segmap;
    depth = single(depth);
    depth = (depth - handPos(3)) / maxDepth;
    depth(msk)=1;
    hmap(repmat(msk,[1 1 lenh]))=0;
    %ir(~segmap)=0;
    bbox = double(bbox);
    
    
    %IR = imtransform(ir,tform,'linear','UData',[1 size(ir,2)], 'VData',[1 size(ir,1)], ...
    %    'XData',[bbox(1) bbox(1)+bbox(3)],'YData',[bbox(2) bbox(2)+bbox(4)], ...
    %    'Size', [imsz imsz]);
    D = imtransform(depth,tform,'linear','UData',[1 size(ir,2)], 'VData',[1 size(ir,1)], ...
        'XData',[bbox(1) bbox(1)+bbox(3)],'YData',[bbox(2) bbox(2)+bbox(4)], ...
        'Size', [imsz imsz]);
    
    D_{1} = D-imfilter(D,h,'replicate');
    D_{2} = imresize(D,[imsz/2 imsz/2]) - imfilter(imresize(D,[imsz/2 imsz/2]),h,'replicate'); 
    D_{3} = imresize(D,[imsz/4 imsz/4]) - imfilter(imresize(D,[imsz/4 imsz/4]),h,'replicate'); 
       
    LBL = imtransform(lbl,tform,'nearest','UData',[1 size(ir,2)], 'VData',[1 size(ir,1)], ...
        'XData',[bbox(1) bbox(1)+bbox(3)],'YData',[bbox(2) bbox(2)+bbox(4)], ...
        'Size', [imsz imsz]);
    HMP = imtransform(hmap,tform,'linear','UData',[1 size(ir,2)], 'VData',[1 size(ir,1)], ...
        'XData',[bbox(1) bbox(1)+bbox(3)],'YData',[bbox(2) bbox(2)+bbox(4)], ...
        'Size', [hsz hsz lenh]);           
    
    
    %IR = reshape(single(IR'),[imsz imsz 1 1]); % THESE ARE TRANSPOSED TO DEAL WITH THE DIFFERENT ROW ORDERING IN CAFFE
    D_{1} = reshape(single(D_{1}'),[imsz imsz 1 1]);
    D_{2} = reshape(single(D_{2}'),[imsz/2 imsz/2 1 1]);
    D_{3} = reshape(single(D_{3}'),[imsz/4 imsz/4 1 1]);
    
    LBL =  reshape(single(LBL'),[imsz imsz 1 1]);
    HMP = reshape(single(permute(HMP,[2 1 3])),[hsz hsz lenh 1]);
    
    Rot = rot(:,:,centerInd); % THIS IS NOT TRANSPOSED BECAUSE I DONT SEE A REASON FOR IT ...
    [~,Angle]=derotate(Rot);
    [Eul(1), Eul(2), Eul(3)] = dcm2angle(Rot);
    Quat = dcm2quat(Rot);
    Thumb = pos(1:3,1)-pos(1:3,centerInd);
    PalmCenter =  pos(1:3,centerInd) - handPos;    
    FullFingers = pos(1:3,(1:5)) - repmat(pos(1:3,centerInd),[1 5]);
    
    imCnt_=mod(imCnt,EXAMPLE_LIM)+1; % ALLOW ONLY EXAMPLELIM examples per file.
    
    if imCnt_==1
        newId = newId + 1; % New file id
    end
    if newId~=oldId % Then create a new set of files
        try
        h5create(sprintf('%s%s%.3i.h5',outDir,'labelQuat_',newId), '/label', [4, inf],'Chunksize',[4 1] );
        h5create(sprintf('%s%s%.3i.h5',outDir,'labelEul_',newId), '/label', [3, inf],'Chunksize',[3 1] );
        h5create(sprintf('%s%s%.3i.h5',outDir,'labelRot_',newId), '/label', [9, inf],'Chunksize',[9 1] );
        h5create(sprintf('%s%s%.3i.h5',outDir,'labelThumb_',newId), '/label', [3, inf],'Chunksize',[3 1] );
        h5create(sprintf('%s%s%.3i.h5',outDir,'labelPalm_',newId), '/label', [3, inf],'Chunksize',[3 1] );
        h5create(sprintf('%s%s%.3i.h5',outDir,'labelFingers_',newId), '/label', [3*5, inf],'Chunksize',[3*5 1] );
        h5create(sprintf('%s%s%.3i.h5',outDir,'labelAngle_',newId), '/label', [1, inf],'Chunksize',[1 1] );
        
        %h5create(sprintf('%s%s%.3i.h5',outDir,'data_D_IR_imsz_',newId), '/data', [imsz,imsz,2,inf], 'Chunksize',[imsz imsz 2 1]);
        h5create(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz,newId), '/data', [imsz,imsz,1,inf], 'Chunksize',[imsz imsz 1 1]);
        h5create(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz/2,newId), '/data', [imsz/2,imsz/2,1,inf], 'Chunksize',[imsz/2 imsz/2 1 1]);
        h5create(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz/4,newId), '/data', [imsz/4,imsz/4,1,inf], 'Chunksize',[imsz/4 imsz/4 1 1]);
        
        h5create(sprintf('%s%s%.3i.h5',outDir,'labelHeatmap_',newId), '/label',  [hsz,hsz,lenh,inf], 'Chunksize',  [hsz hsz lenh 1]);
        h5create(sprintf('%s%s%.3i.h5',outDir,'labelClass_',newId), '/label',  [imsz,imsz,1,inf], 'Chunksize',  [imsz imsz 1 1]);
                
        %h5create(sprintf('%s%s%.3i.h5',outDir,'data_IR_imsz_',newId), '/data', [imsz,imsz,1,inf], 'Chunksize',[imsz imsz 1 1]);
        catch e
            fprintf('Creating databases error: probably already exist\n');
        end
        oldId = newId;
    end
    
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelQuat_',newId), '/label', Quat(:),[1 imCnt_] ,[4 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelEul_',newId), '/label',Eul(:),[1 imCnt_] ,[3 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelRot_',newId), '/label',Rot(:),[1 imCnt_] ,[9 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelThumb_',newId), '/label',Thumb(:),[1 imCnt_], [3 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelPalm_',newId), '/label',PalmCenter(:),[1 imCnt_], [3 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelFingers_',newId), '/label',FullFingers(:),[1 imCnt_] ,[3*5 1] );
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelAngle_',newId), '/label',Angle(:),[1 imCnt_] ,[1 1] );
    
    %h5write(sprintf('%s%s%.3i.h5',outDir,'data_D_IR_imsz_',newId), '/data', cat(3,D,IR) ,[1 1 1 imCnt_], [imsz imsz 2 1]);
    h5write(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz,newId), '/data', D_{1} ,[1 1 1 imCnt_],[imsz imsz 1 1]);
    h5write(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz/2,newId), '/data', D_{2} ,[1 1 1 imCnt_],[imsz/2 imsz/2 1 1]);
    h5write(sprintf('%s%s%i_%.3i.h5',outDir,'data_D_',imsz/4,newId), '/data', D_{3} ,[1 1 1 imCnt_],[imsz/4 imsz/4 1 1]);
    
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelHeatmap_',newId), '/label', HMP ,[1 1 1 imCnt_],[hsz hsz lenh 1]);
    h5write(sprintf('%s%s%.3i.h5',outDir,'labelClass_',newId), '/label', LBL ,[1 1 1 imCnt_],[imsz imsz 1 1]);
    
    %h5write(sprintf('%s%s%.3i.h5',outDir,'data_IR_imsz_',newId), '/data', IR ,[1 1 1 imCnt_],[imsz imsz 1 1]);
    
    imCnt = imCnt + 1;
    % ---------------------------------------------------------------------------------
    % still need to remove the mean from each image and perform melin or something else
    % ---------------------------------------------------------------------------------
    %cla
    %imagesc(depth)
    %rectangle('Position',bbox);
    %pause
    drawnow
end
clear OKAY

error('i.e. you still need to run or edit and run preparefinaldata.m and make sure the disk has enough space first');
