function showcirclefeaturesrad(pos,fcol,lwidth,gcah)

%
% showcirclefeatures(pos,fcol,lwidth,gcah)
%
%  Adds circles to the current plot with 'pos' Nx3 matrix
%  specifying (by columns) the center of circles x,y and
%  the radius of circles
%

if nargin<2 fcol=[0 0 1]; end
if nargin<3 lwidth=1; end
if nargin<4 gcah=gca; end

if size(pos,2)==2 pos=[pos 5*ones(size(pos,1),1)]; end
a=transpose(linspace(0,2.1*pi,21));
points=[cos(a) sin(a)];
if length(pos)>2000
    warning(['features are too dense, plotting centers only. Scales are ' num2str( unique(pos(:,3))')]);
end
set(gcah,'NextPlot','add')
for i=1:size(pos,1)
  x0=pos(i,1); y0=pos(i,2); r0=pos(i,3);
  cpoints=points*r0;
  cpoints(:,1)=cpoints(:,1)+x0;
  cpoints(:,2)=cpoints(:,2)+y0;
  if size(fcol,1)>1 
      col=fcol(i,:);
  else
      col=fcol;
  end
  if (length(pos)<2000)
      h=plot(cpoints(:,1),cpoints(:,2),'LineWidth',lwidth,'Parent',gcah);
      set(h,'Color',col)
  end
  h=plot(x0,y0,'x','Parent',gcah);
  set(h,'Color',col)
end

drawnow
