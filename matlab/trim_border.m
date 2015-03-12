function output_image = trim_border(input_image, num_pixels)
%by James Hays

%removes num_pixels from each edge of the image, returning a smaller image
%different from remove border, which just recolors the border and doesn't
%resize the image.

output_image = input_image(1+num_pixels:end-num_pixels, 1+num_pixels:end-num_pixels, :);