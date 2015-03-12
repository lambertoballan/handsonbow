% function [] = detect_features(im_dir);
%
% Detect and describe features for all images in a directory 
%
% IN: im_dir ... directory of images (assumed to be *.jpg)
% OUT: for each image, a matlab file *.desc is created in directory im_dir, 
%      containing detected LoG features described with SIFT descriptors.
%
% The output Matlab file contains structure "desc" with fileds:
%
%  desc.r    ... row index of each feature
%  desc.c    ... column index of each feature
%  desc.rad  ... radius (scale) of each feauture
%  desc.sift ... 128d SIFT descriptor for each feature
%
% Josef.Sivic@ens.fr
% 7/11/2009
%
% Modified by Lamberto Ballan - 2/9/2013

function [] = detect_features(im_dir,file_ext,show_img)
        
    dd = dir(fullfile(im_dir,'*.jpg'));
    if ~exist('show_img','var')
        show_img = false;
    end    

    % detector paramteres
    sigma       = 2;              % initial scale
    k           = sqrt(sqrt(2));  % scale step
    sigma_final = 16;             % final scale
    threshold   = 0.005;          % squared response threshold

    % descriptor parameters
    enlarge_factor = 2; % enlarge the size of the features to make them more distinctive

    parfor i = 1:length(dd)
        %fname = [im_dir,'/',dd(i).name];
        fname = fullfile(im_dir,dd(i).name);

        fname_out = [fname(1:end-3),file_ext];
        if exist(fname_out,'file')
            fprintf('File exists! Skipping %s \n',fname_out);
            continue;
        end;

        fprintf('Detecting and describing features: %s \n',fname_out);
        Im = imread(fname);
        Im = mean(double(Im),3)./max(double(Im(:)));

        % compute features (LoG)
        [r, c, rad] = BlobDetector(Im, sigma, k, sigma_final, threshold);

        % describe features
        circles = [c r rad];
        sift_arr = find_sift(Im, circles, enlarge_factor);

        % convert to single to save disk space
        desc = struct('sift',uint8(512*sift_arr),'r',r,'c',c,'rad',rad);

        iSave(desc,fname_out);

        if show_img
            d=desc;
            Im = imread(fname);
            figure; clf, showimage(Im);
            x=d.c;
            y=d.r;
            rad=d.rad;
            showcirclefeaturesrad([x,y,rad]);
            pause
        end
    end

end

function iSave(desc,fName)
    save(fName,'desc');
end






