function [img,x,y]=combimage(imglist,ccol,aspectratio);

if nargin<2 ccol=[]; end
if nargin<3 aspectratio=1; end

if ~iscell(imglist) imglist=cell(imglist); end
n=length(imglist);


nc=ceil(aspectratio*sqrt(n));
nr=ceil(n/nc);

xsize=1000;
ysize=round(xsize*nr/nc);
img=256*ones(ysize,xsize,3);
xpos=round(linspace(1,xsize,nc+1));
ypos=round(linspace(1,ysize,nr+1));

k=1;
for i=1:nr
  for j=1:nc
    if k<=n
      xsz=xpos(j+1)-xpos(j)+1;
      ysz=ypos(i+1)-ypos(i)+1;
      imgsample=imglist{k};
      if size(imgsample,3)==1;
	imgsample(:,:,2)=imgsample(:,:,1);
	imgsample(:,:,3)=imgsample(:,:,1);
      end
      [yszin,xszin,cszin]=size(imgsample);
      if (xszin/xsz)>(yszin/ysz) xszout=xsz; xoff=0; yszout=round(yszin*xsz/xszin); yoff=floor((ysz-yszout)/2); end
      if (xszin/xsz)<=(yszin/ysz) yszout=ysz; yoff=0; xszout=round(xszin*ysz/yszin); xoff=floor((xsz-xszout)/2); end
      img(yoff+ypos(i):yoff+(ypos(i)+yszout-1),xoff+xpos(j):xoff+(xpos(j)+xszout-1),:)=resizeimage(imgsample,xszout,yszout);
      x(k)=xoff+xpos(j);
      y(k)=yoff+ypos(i);      
      if size(ccol,1)>=k % draw color marker
	img(yoff+ypos(i):yoff+ypos(i)+10,xoff+xpos(j):xoff+xpos(j)+20,1)=ccol(k,1)*256;
	img(yoff+ypos(i):yoff+ypos(i)+10,xoff+xpos(j):xoff+xpos(j)+20,2)=ccol(k,2)*256;
	img(yoff+ypos(i):yoff+ypos(i)+10,xoff+xpos(j):xoff+xpos(j)+20,3)=ccol(k,3)*256;
      end
      k=k+1;
    end
  end
end
