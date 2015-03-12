function y = log2space(d1, d2, n)
%LOGSPACE Logarithmically spaced vector.
%   LOGSPACE(X1, X2) generates a row vector of 50 logarithmically
%   equally spaced points between decades 2^X1 and 2^X2.  
%
%   LOGSPACE(X1, X2, N) generates N points.
%   For N < 2, LOGSPACE returns 2^X2.
%
%   Class support for inputs X1,X2:
%      float: double, single
%
%   See also LINSPACE, :.

%   Copyright 1984-2005 The MathWorks, Inc. 
%   $Revision: 5.11.4.2 $  $Date: 2005/10/03 16:12:49 $

if nargin == 2
    n = 50;
end
n = double(n);


y = (2).^ [d1+(0:n-2)*(d2-d1)/(floor(n)-1), d2];
