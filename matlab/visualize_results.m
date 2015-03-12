function [ ] = visualize_results( classes, desc_test, labels_test, labels_result )

%VISUALIZE_EXAMPLES Illustrate correcly classified and missclassified 
%samples of each class.

figure;

for i=1:length(classes)
    
    ind=find(labels_test==i);
    %bof_chi2acc_class=mean(result_labels(ind)==labels_test(ind));
    indcorr=ind(find(labels_result(ind)==labels_test(ind)));
    indmiss=ind(find(labels_result(ind)~=labels_test(ind)));

    clf
    imgcorr={};
    if length(indcorr)
        for j=1:length(indcorr) 
            imgcorr{end+1}=imread(desc_test(indcorr(j)).imgfname);
        end
        subplot(1,2,1), showimage(combimage(imgcorr,[],1))
        title(sprintf('%d Correctly classified %s images',length(indcorr),classes{i}))
    end
    
    imgmiss={};
    if length(indmiss)
        for j=1:length(indmiss)
            imgmiss{end+1}=imread(desc_test(indmiss(j)).imgfname);
        end
        subplot(1,2,2), showimage(combimage(imgmiss,[],1))
        title(sprintf('%d Miss-classified %s images',length(indmiss),classes{i}))
    end
    
    pause;
end

end

