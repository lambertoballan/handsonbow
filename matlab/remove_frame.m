function [output] = remove_frame(input_image)
%by James Hays

%removes the solid color border around images, such as those commonly used
%on flickr.  Has a preference towards removing white and black borders as
%well (which most of them are)

%I think this function expects image to be in the [0,1] range, and double
%precision

% if(size(input_image,1) < min_resolution || size(input_image,2) < min_resolution)
% %     fprintf('error, image too small\n');
%     output = input_image;
%     return
% end

input_copy = input_image;

if(size(input_image,3) == 3)
    input_image = my_rgb2gray(input_image);
end

visualization_copy = input_image;

color_tolerance = .04; %the amount of variance allowed for something to be considered border

border_width = 1;
max_border_width = round(max(size(input_image))/8);
keep_going = 1;

scores = zeros(1, max_border_width);

while( border_width <= max_border_width && keep_going == 1)
%     fprintf('testing border = %d pixels', border_width);

    %lets cut out and vectorize the regions, who cares if we double count
    %the corners for now.
    
%     top    = input_image(1:border_width, :);
%     bottom = input_image(end-border_width:end, :);
%     left   = input_image(:,1:border_width);
%     right  = input_image(:,end-border_width:end);
    
    top    = input_image(border_width, :);
    bottom = input_image(end-border_width+1, :);
    left   = input_image(:,border_width);
    right  = input_image(:,end-border_width+1);
    
    all_pixels = [top(:);bottom(:);left(:);right(:)];
    
    average_intensity = mean(all_pixels);
    std_intensity     = std(all_pixels);
    
    intensity_score = min(1-average_intensity, average_intensity);
    std_score       = std_intensity;
    
    borderness = std_score * intensity_score;
    scores(border_width) = borderness;
    
%     figure(1)
%     imshow(trim_border(input_copy, border_width))
%     pause(.1)
    
    border_width = border_width + 1;
end

% figure(1)
% plot(scores)

last_border_column = max(find(scores < .005));
if(isempty(last_border_column))
    output = input_copy;
    last_border_column = 0;
else
    %lets trim a little extra just to make sure, since
    %many times the borders are layered and concentric
    last_border_column = last_border_column + 12;
    output = trim_border(input_copy, last_border_column);
end

fprintf('   trimmed %.4d pixels \n', last_border_column);

% figure(2)
% imshow(output)