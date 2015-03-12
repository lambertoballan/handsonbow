% function chi = chi2(h,g)
% 
% Compute Chi2 statistics between two L1 normalized vectors, h and g.
function chi = chi2(h,g)

  h = h./(sum(h)+eps);
  g = g./(sum(g)+eps);
  chi = 0.5*sum(((h-g).^2)./(h+g+eps));

return;
