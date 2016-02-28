function patches = createPatches(I, patchSize, overlap, onePatch)
% This takes an image and a patchsize and extracts all the patches
% from that are possible with or without overlap

[rows cols chan] = size(I);

if onePatch
    numRowPatches = 1;
    numColPatches = 1;
    rowDelta = floor((rows - patchSize(1))/numRowPatches);
    colDelta = floor((cols - patchSize(2))/numColPatches);
else
    if overlap
        numRowPatches = floor(rows/patchSize(1));
        numColPatches = floor(cols/patchSize(2));
        rowDelta = floor((rows - patchSize(1))/numRowPatches);
        colDelta = floor((cols - patchSize(2))/numColPatches);
    else
        numRowPatches = floor(rows/patchSize(1))-1;
        numColPatches = floor(cols/patchSize(2))-1;
        rowDelta = patchSize(1);
        colDelta = patchSize(2);
    end
end
cnt = 1;
rowInds = 1:patchSize(1);
colInds = 1:patchSize(2);

for u=0:rowDelta:(numRowPatches)*rowDelta
    for v = 0:colDelta:(numColPatches)*colDelta
        patches(:,:,:,cnt)  = I( u + rowInds,  v + colInds, :);
        cnt = cnt + 1;
    end
end
