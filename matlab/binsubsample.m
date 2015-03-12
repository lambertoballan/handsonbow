function pixels = binsubsample(inpic,filtermask,nsub)
% BINSUBSAMPLE -- subsampling with binomial presmoothing
%
%   binsubsample(image) reduces the size of an image by first smoothing
%   it with a two-dimensional binomial kernel having filter coefficients 
%      (1/16  1/8  1/16)
%      ( 1/8  1/4  1/8)
%      (1/16  1/8  1/16)
%   and then subsampling it by a factor of two in each dimension.

%
% Check of input arguments turned off -- has surprising side effects!
%
% if ((nargin ~= 1) | (isempty(image)))
%   error('One non-empty matrix must be given as input')
% return
%


if nargin<2 filtermask=[]; end
if nargin<3 nsub=1; end

if ~length(filtermask) prefilterrow = [1 2 1]/4;
else prefilterrow=filtermask;
end

prefilter = prefilterrow' * prefilterrow;
pixels=inpic;

for j=1:nsub
  presmoothpic=[];
  for i=1:size(inpic,3)
    presmoothpic(:,:,i) = crop2(filter2(prefilter, extend2(pixels(:,:,i),2,2)),2,2);
  end
  pixels = rawsubsample(presmoothpic);
end

if 0 % original code
  if nargin>1 prefilterrow=filtermask;
  else prefilterrow = [1 2 1]/4;
  end
  
  prefilter = prefilterrow' * prefilterrow;
  presmoothpic=[];
  for i=1:size(inpic,3)
    %presmoothpic(:,:,i) = filter2(prefilter, inpic(:,:,i));
    presmoothpic(:,:,i) = crop2(filter2(prefilter, extend2(inpic(:,:,i),2,2)),2,2);
  end
  pixels = rawsubsample(presmoothpic);
end