% function [] = detect_features_uva_csift(im_dir);
%
% Detect and describe features for all images in a directory 
%
% IN: im_dir ... directory of images (assumed to be *.jpg)
% OUT: for each image, a matlab file *.desc_hess is created in directory im_dir, 
%      containing the Hess' SIFT  Pyramid descriptors.
%
% The output Matlab file contains structure "desc" with fileds:
%
%  desc.r    ... row index of each feature
%  desc.c    ... column index of each feature
%  desc.rad  ... radius (scale) of each feauture
%  desc.sift ... Pyramid SIFT descriptor for each feature
%
%

function detect_features_gist(im_dir,file_ext,varargin)

    
	
    dd = dir([im_dir,'/*.jpg']);

    for i = 1:length(dd)
        fname = [im_dir,'/',dd(i).name];  
        I=imread(fname);
        fname_out = [fname(1:end-3),file_ext];

        if exist(fname_out,'file')
            fprintf('File exists! Skipping %s \n',fname_out);
            continue;
        end;
        
        fprintf('Detecting and describing features (gist): %s \n',fname_out);

        %gist params
        Nblocks = 4;
        imageSize = 256; 
        orientationsPerScale = [8 8 4];
        numberBlocks = 4;
        
        
        if(size(I,3)<3)
           I=repmat(I,[1 1 3]);
        end
        %if size(I,1) ~= imageSize
           I = imresize(I, [imageSize imageSize], 'bilinear');
        %end
        % Precompute filter transfert functions (only need to do this one, unless image size is changes):
        %createGabor(orientationsPerScale, imageSize); % this shows the filters
        G = createGabor(orientationsPerScale, imageSize);

        % Computing gist requires 1) prefilter image, 2) filter image and collect
        % output energies
        output = prefilt(double(I), 4);
        g = gistGabor(output, numberBlocks, G);
        
        desc = struct('g',g);

        iSave(desc,fname_out);

    end

end


function iSave(desc,fName)
    save(fName,'desc');
end

