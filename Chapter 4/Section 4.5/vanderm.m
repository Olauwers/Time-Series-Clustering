function varargout = vanderm(v,c)
%VANDERM Vandermonde matrix
%  Creating matrix with terms of geometric progression in each row
%  Extending the vander.m of Matlab
%
%  A = vanderm(v,c) returns the n*c Vandermonde matrix whose columns are
%  powers of the vector v, i.e. A(i,j) = v(i)^(j-1), where n is the length
%  of v. If c is not specified, square matrix is assumed, i.e. c=n.
%
%  [A,D] = vanderm(v,c) also returns the determinant of square matrix A.
%
% Siqing Wu, <6sw21@queensu.ca> 
% Version: 1.0, Date: 2008-07-29

if all(size(v)>1)
    error('Input v should be a vector.')
end
error(nargchk(1,2,nargin))
error(nargoutchk(0,2,nargout))

v = v(:); % make v a column
n = length(v);
if nargin < 2
    c = n; 
end
X = [ones(n,1) repmat(v,[1 c-1])];
A = cumprod(X,2);

switch nargout
    case {0,1}
        varargout = {A};
    case 2
        if ~(n==c)  % determinant only for square matrix
            disp('Error: Matrix is not square! Determinant is not defined...')
            varargout = {A, []};
        else
            % instead of det(A) which takes a long time when A is large
            Y = repmat(v(end:-1:2).',[c-1 1]) - repmat(v(end-1:-1:1),[1 c-1]);
            L = Y(logical(tril(ones(c-1))));
            D = prod(L);
            varargout = {A, D};
        end
end

