%% Split an HDF5 file into multiple files

if ~exist('OKAY')
    error('make sure disk has enough space first. Also check the following params.');
    error('Also change filebase according to whether it is derotated or note');
end

numFiles = 10; % This must be equal to the number of files/20k
writeTrain = 1;
writeTest = 0;
fileBase = sprintf('');

ratio = 0; %Percent of data to use for test
imsz = 96;

%outExpName = 'FinalDataNYUTestHDF5_HM/';
%dataDir = ['/home/gipuser/Documents/' outExpName];
dataDir = outDir;
outBase =  sprintf('data_%i_',ratio);
fileListDir = '/home/gipuser/caffe-workspace/caffe_new2/examples/handnet/';



%%
fprintf('----------------------- Writing ----------------------------\n');
for fileCnt=0:numFiles-1
    %% Read data
    tic
    clear A HM C D Palm Thmb Fng Q E R
    Q=single(h5read(sprintf('%s%s%.3i.h5',dataDir,'labelQuat_',fileCnt), '/label')); fprintf('Read Q\n');
    E=single(h5read(sprintf('%s%s%.3i.h5',dataDir,'labelEul_',fileCnt), '/label')); fprintf('Read E\n');
    R=single(h5read(sprintf('%s%s%.3i.h5',dataDir,'labelRot_',fileCnt), '/label')); fprintf('Read R\n');
    Thmb=single(h5read(sprintf('%s%s%.3i.h5',dataDir,'labelThumb_',fileCnt), '/label')); fprintf('Read Thmb\n');
    Palm=single(h5read(sprintf('%s%s%.3i.h5',dataDir,'labelPalm_',fileCnt), '/label')); fprintf('Read Palm\n');
    Fng=single(h5read(sprintf('%s%s%.3i.h5',dataDir,'labelFingers_',fileCnt), '/label')); fprintf('Read Fng\n');
    A=single(h5read(sprintf('%s%s%.3i.h5',dataDir,'labelAngle_',fileCnt), '/label')); fprintf('Read Angle\n');
    
    D_{1}  =single(h5read(sprintf('%s%s%i_%.3i.h5',dataDir,'data_D_',imsz,fileCnt), '/data')); fprintf('Read D96\n');
    D_{2}  =single(h5read(sprintf('%s%s%i_%.3i.h5',dataDir,'data_D_',imsz/2,fileCnt), '/data')); fprintf('Read D48\n');
    D_{3}  =single(h5read(sprintf('%s%s%i_%.3i.h5',dataDir,'data_D_',imsz/4,fileCnt), '/data')); fprintf('Read D24\n');
    
    HM  =single(h5read(sprintf('%s%s%.3i.h5',dataDir,'labelHeatmap_',fileCnt), '/label')); fprintf('Read HM \n');
    C  =single(h5read(sprintf('%s%s%.3i.h5',dataDir,'labelClass_',fileCnt), '/label')); fprintf('Read Class\n');
    
    sz = [imsz imsz 1 length(Q)];
    
    % No need to shuffle because writehdf5data has done it for us (hopefully
    % the flag is on there)
    
    %% Write out data
    
    split = uint32(ratio/100*length(Q)); % Take ratio% point as start of train data
    if split==0
        split=1;
    end
    % Train and test data
    
    % ---------------------------------Standard D and R but without normalization------------------------------------------------
    D=D_{1};
    if writeTrain
        h5create(sprintf('%s%s%s_%.3i.h5',dataDir,outBase,'train',fileCnt),'/data',size( D(:,:,:,split:end)));
        h5write(sprintf('%s%s%s_%.3i.h5',dataDir,outBase,'train',fileCnt),'/data',       D(:,:,:,split:end));fprintf('Written train data\n');
        h5create(sprintf('%s%s%s_%.3i.h5',dataDir,outBase,'train',fileCnt),'/label',size(R(:,split:end)));
        h5write(sprintf('%s%s%s_%.3i.h5',dataDir,outBase,'train',fileCnt),'/label',      R(:,split:end));fprintf('Written train label\n');
    end
    if writeTest
        h5create(sprintf('%s%s%s_%.3i.h5',dataDir,outBase,'test',fileCnt),'/data',size(  D(:,:,:,1:split-1)));
        h5write(sprintf('%s%s%s_%.3i.h5',dataDir,outBase,'test',fileCnt),'/data',        D(:,:,:,1:split-1));fprintf('Written test data\n');
        h5create(sprintf('%s%s%s_%.3i.h5',dataDir,outBase,'test',fileCnt),'/label',size( R(:,1:split-1)));
        h5write(sprintf('%s%s%s_%.3i.h5',dataDir,outBase,'test',fileCnt),'/label',       R(:,1:split-1));fprintf('Written test label\n');
    end
    % ------------------------------------------------------------------------------------------------------
    
    % ---------------------------------Heatmap------------------------------------------------
    for j=1:3
        if writeTrain
            h5create(sprintf('%s%s%s%i_%.3i.h5',dataDir,outBase,'trainHM',imsz/2^(j-1),fileCnt),'/data',size( D_{j}(:,:,:,split:end)));
            h5write(sprintf('%s%s%s%i_%.3i.h5',dataDir,outBase,'trainHM',imsz/2^(j-1),fileCnt),'/data',       D_{j}(:,:,:,split:end));fprintf('Written trainHM_%i data\n',imsz/2^(j-1));
            if j==3,  hm = R(:,split:end); %96
            elseif j==2, hm = A(:,split:end); %48
            else hm = reshape(HM(:,:,:,split:end),hsz*hsz*5,[]); %24
            end
            h5create(sprintf('%s%s%s%i_%.3i.h5',dataDir,outBase,'trainHM',imsz/2^(j-1),fileCnt),'/label',size( hm));
            h5write(sprintf('%s%s%s%i_%.3i.h5',dataDir,outBase,'trainHM',imsz/2^(j-1),fileCnt),'/label',       hm);fprintf('Written labelHM data\n');
        end
        
        if writeTest
            h5create(sprintf('%s%s%s%i_%.3i.h5',dataDir,outBase,'testHM',imsz/2^(j-1),fileCnt),'/data',size(  D_{j}(:,:,:,1:split-1)));
            h5write(sprintf('%s%s%s%i_%.3i.h5',dataDir,outBase,'testHM',imsz/2^(j-1),fileCnt),'/data',        D_{j}(:,:,:,1:split-1));fprintf('Written testHM_%i data\n',imsz/2^(j-1));
            if j==3,  hm = R(:,1:split-1); %96
            elseif j==2, hm = A(:,1:split-1); %48
            else hm = reshape(HM(:,:,:,1:split-1),hsz*hsz*5,[]); %24
            end
            h5create(sprintf('%s%s%s%i_%.3i.h5',dataDir,outBase,'testHM',imsz/2^(j-1),fileCnt),'/label',size(  hm));
            h5write(sprintf('%s%s%s%i_%.3i.h5',dataDir,outBase,'testHM',imsz/2^(j-1),fileCnt),'/label',        hm);fprintf('Written testHM label\n');
        end
    end
    %------------------------------------------------------------------------------------------------------
    
    
    fprintf('Completed file #%i in %f\n',fileCnt,toc);
end

%% 
if writeTrain
    ftrain=fopen(sprintf('%sTrainFiles%s.txt',fileListDir,fileBase),'wt');
end

if writeTest
    ftest=fopen(sprintf('%sTestFiles%s.txt',fileListDir,fileBase),'wt');
end

for j=1:3
    if writeTrain
        ftrainhm{j}=fopen(sprintf('%sTrainFilesHM_%i_18x18%s.txt',fileListDir,imsz/2^(j-1),fileBase),'wt');
    end
    if writeTest
        ftesthm{j}=fopen(sprintf('%sTestFilesHM_%i_18x18%s.txt',fileListDir,imsz/2^(j-1),fileBase),'wt');
    end
end

for fileCnt=0:numFiles-1
    if writeTrain
        fwrite(ftrain,sprintf('%s%s%s_%.3i.h5\n',dataDir,outBase,'train',fileCnt));
    end
    if writeTest
        fwrite(ftest,sprintf('%s%s%s_%.3i.h5\n',dataDir,outBase,'test',fileCnt));
    end
    
    for j=1:3
        if writeTrain
            fwrite(ftrainhm{j},sprintf('%s%s%s%i_%.3i.h5\n',dataDir,outBase,'trainHM',imsz/2^(j-1),fileCnt));
        end
        if writeTest
            fwrite(ftesthm{j}, sprintf('%s%s%s%i_%.3i.h5\n',dataDir,outBase,'testHM',imsz/2^(j-1),fileCnt));
        end
    end
end

fclose all
%%
