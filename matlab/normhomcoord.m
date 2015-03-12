function vn=normhomcoord(v)

% vn=normhomcoord(v)
%   v: input is a m*n matrix with n vectors with homogeneous coordinates
%  vn: output is a marix of corresponding vectors with normalized homogeneous coordiates




ind=find(abs(v(end,:))<eps);
v(end,ind)=eps;

vn=v./(ones(size(v,1),1)*v(end,:));
