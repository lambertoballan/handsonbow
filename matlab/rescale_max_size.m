function output_image = rescale_max_size(input_image, max_allowed_size, verbose)
%
% written by Lamberto Ballan <www.lambertoballan.net>, October 2011
% (originally written by James Hays)

%if all image dimensions are less than max_allowed_size, this function does
%nothing.  Otherwise it will scale the image down, uniformly, so that the
%largest image dimension has size max_allowed_size

%works for color or grayscale

if exist('verbose') == 1
    ;
else
    verbose = 0;
end

if(size(input_image,1) > max_allowed_size || size(input_image,2) > max_allowed_size)
    %1 if 1 is larger, 2 if 2 is larger
    max_dimension = (size(input_image,2) > size(input_image,1)) + 1;

    scale_factor = max_allowed_size/size(input_image,max_dimension);

    if(verbose)
        fprintf('   Resizing image from %dx%d to %dx%d\n',...
            size(input_image,2), size(input_image,1), ...
            round(size(input_image,2) * scale_factor), ...
            round(size(input_image,1) * scale_factor));
    end
    output_image = imresize(input_image, scale_factor, 'bilinear');
else
    if(verbose)
        fprintf('   Image does not need to be rescaled (%dx%d)\n',size(input_image,2), size(input_image,1));
    end
    output_image = input_image;
end
