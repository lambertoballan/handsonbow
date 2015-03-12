function result=extend2(data,ny,nx)

  
%[ysz,xsz,csz]=size(data);
%result=cropbbox(data,[-nx -ny xsz+nx ysz+ny]);

if 1 % older version  
  [ysize,xsize]=size(data);
  newxsize=xsize+2*nx;
  newysize=ysize+2*ny;
  result=zeros(newysize,newxsize);
  result(ny+1:ysize+ny,nx+1:xsize+nx)=data;
  
  for x=1:nx result(:,x)=result(:,nx+1); end
  for x=xsize+nx+1:newxsize result(:,x)=result(:,xsize+nx); end
  
  for y=1:ny result(y,:)=result(ny+1,:); end
  for y=ysize+ny+1:newysize result(y,:)=result(ysize+ny,:); end
end