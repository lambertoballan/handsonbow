function imgout=cropbbox(img,bbox,missingpixflag)

% imgout=cropbbox(img,bbox)
% 
% crop posibbly multi-chanel image by a
% bounding box bbox=[xmin,ymin,xmax,ymax].
% If the bounding box exceeds image dimensions
% the image is replicated by mirroring image
% pixels outside image area.

if nargin<3 missingpixflag=0; end

bbox=round(bbox);
[ysz,xsz,csz]=size(img);

if bbox(1)>=1 & bbox(3)<=xsz & ...
   bbox(2)>=1 & bbox(4)<=ysz

  imgout=img(bbox(2):bbox(4),bbox(1):bbox(3),:);
else
  
  % fill unknown pixels with the mirrored image
  [x,y]=meshgrid(bbox(1):bbox(3),bbox(2):bbox(4));
  xorig=x; yorig=y;
  go=1;
  while go
    go=0;
    ix=find(x(:)<1); iy=find(y(:)<1);
    if missingpixflag>=1                            % repeat border pixels 
      if length(ix) x(ix)=1; end
      if length(iy) y(iy)=1; end
    else                                            % mirror border pixels 
      if length(ix) x(ix)=abs(x(ix))+2; end
      if length(iy) y(iy)=abs(y(iy))+2; end
    end
    ix=find(x(:)>xsz); iy=find(y(:)>ysz); 
    if missingpixflag>=1                            % repeat border pixels 
      if length(ix) x(ix)=xsz; go=1; end
      if length(iy) y(iy)=ysz; go=1; end
    else                                            % mirror border pixels 
      if length(ix) x(ix)=2*xsz-x(ix)-1; go=1; end
      if length(iy) y(iy)=2*ysz-y(iy)-1; go=1; end
    end
  end
  
  % handle n-dimensional images
  szorig=size(img);
  img=reshape(img,xsz*ysz,csz);
  ind=sub2ind([ysz,xsz],y(:),x(:));
  sz=size(x);
  if length(szorig>2) sz=[sz szorig(3:end)]; end

  % compose cropped image
  imgout=img(ind,:);
  
  if missingpixflag==2 % set missing pixels to zero
    indmissing=find(xorig(:)<1 | xorig(:)>xsz | yorig(:)<1 | yorig(:)>ysz);
    imgout(indmissing,:)=0;
  end
  
  imgout=reshape(imgout,sz);
end