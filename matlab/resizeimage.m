function imgout=resizeimage(imgin,szxout,szyout)
  
szxout=max(1,szxout);
szyout=max(1,szyout);

[szyin,szxin,szcin]=size(imgin);
sx=szxout/szxin;
sy=szyout/szyin;



%keyboard
if sx==1 & sy==1
  % do nothig
  imgout=imgin;
elseif sx==.5 & sy==.5
  %fprintf('Subsampling\n')
  imgout=binsubsample(imgin);
elseif sx==.25 & sy==.25
  imgout=binsubsample(binsubsample(imgin));
elseif sx<.5 & sy<.5
  imgout=resizeimage(binsubsample(imgin),szxout,szyout);
else
  H=[sx 0 0; 0 sy 0; 0 0 1];
  imgout=warpimagehomography(imgin,H,1);
end