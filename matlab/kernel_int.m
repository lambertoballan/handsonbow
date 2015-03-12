function sim = kernel_int(x,y)
    
    sim = zeros(size(x,1),size(y,1));
    for i=1:size(x,1)
        for j=1:size(y,1)
            hists = [x(i,:);y(j,:)];
            sim(i,j) = sum(min(hists));
        end
    end

end
