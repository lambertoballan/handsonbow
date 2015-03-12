function imgout=warpimagehomography(imgin,H,changesizeflag)

% imout=warpimagehomography(imgin,H,changesizeflag)
%

if nargin<3 changesizeflag=0; end
 
[szy,szx,csz]=size(imgin);

% generate coordinates for all points in imgout
if ~changesizeflag
  [x,y]=meshgrid(1:szx,1:szy);
else
  crnrs=[0,0,szx,szx; 0,szy,0,szy; 1 1 1 1];
  crnrstr=normhomcoord(H*crnrs);
  [x,y]=meshgrid(round(min(crnrstr(1,:))+1):round(max(crnrstr(1,:))),...
		 round(min(crnrstr(2,:))+1):round(max(crnrstr(2,:))));
end
points=[transpose(x(:)); transpose(y(:)); ones(1,length(x(:)))];
points_map=normhomcoord(inv(H)*points);

Xfloor=floor(points_map(1,:)); Xplus=points_map(1,:)-Xfloor;
Yfloor=floor(points_map(2,:)); Yplus=points_map(2,:)-Yfloor;
clear points_map points

W{1}=abs((1-Xplus).*(1-Yplus)); X{1}=Xfloor;   Y{1}=Yfloor;
W{2}=abs(  (Xplus).*(1-Yplus)); X{2}=Xfloor+1; Y{2}=Yfloor;
W{3}=abs((1-Xplus).*  (Yplus)); X{3}=Xfloor;   Y{3}=Yfloor+1;
W{4}=abs(  (Xplus).*  (Yplus)); X{4}=Xfloor+1; Y{4}=Yfloor+1;

imgout=zeros([size(x) csz]);
for ic=1:csz
  imgo=zeros(prod(size(x)),1);
  imgc=imgin(:,:,ic);
  for i=1:4
    X{i}(find(X{i}(:)<1))=1; X{i}(find(X{i}(:)>szx))=szx;
    Y{i}(find(Y{i}(:)<1))=1; Y{i}(find(Y{i}(:)>szy))=szy;
    
    ind=sub2ind([szy szx],Y{i}(:),X{i}(:));
    %size(imgo)
    %size(W{i})
    %size(ind)
    imgo=imgo+double(imgc(ind)).*double((W{i}(:)));
  end
  imgo=reshape(imgo,size(x));
  imgout(:,:,ic)=imgo;
end

if 0 % old part
xmap=round(points_map(1,:));
ymap=round(points_map(2,:));

% identify valid coordinates
mask=(xmap<1)+(xmap>szx)+(ymap<1)+(ymap>szy);
valind=find(~mask);

grayvalind=sub2ind(size(imgin),ymap(valind),xmap(valind));

for i=1:csz
  imgout(valind+(i-1)*szx*szy)=imgin(grayvalind+(i-1)*szx*szy);
end
end
