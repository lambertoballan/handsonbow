function extract_gist_features(dirname,file_ext)

% e.g. Caltech-101
% dirname = '../img/101_ObjectCategories';
% file_ext = 'sift';

dirname = ['..' dirname '/'];

d = dir(dirname);
d = d(3:end); %remove . and .. 
for i=1:length(d)
    %detect_features_pyramid_hess(['../img/101_ObjectCategories/' d(i).name],'dense_hess_ori',true);
    detect_features_gist([dirname d(i).name],file_ext);
end

end