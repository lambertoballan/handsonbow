function pixels = rawsubsample(inpic)
% RAWSUBSAMPLE -- reduce image size by raw subsampling without presmoothing
%
%   rawsubsample(image) reduces the size of an image by a factor of two in
%   each dimension by raw subsampling, i.e., by picking out every second
%   pixel along each dimension.

%
% Check of input arguments turned off -- has surprising side effects!
%
% if ((nargin ~= 1) | (isempty(image)))
%   error('One non-empty matrix must be given as input')
%   return
% end
%
 
[m, n, c] = size(inpic);
pixels = inpic(1:2:m, 1:2:n, :);
