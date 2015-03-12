function result=crop2(data,ny,nx)

[ysize,xsize]=size(data);
result=data(ny+1:ysize-ny,nx+1:xsize-nx);
