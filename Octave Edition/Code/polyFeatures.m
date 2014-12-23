function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (m*n matrix) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x n) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%
[m,n]=size(X);

% You need to return the following variables correctly.
X_poly = zeros(m, n*p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

for j=0:n-1
	for i=1:p
    	X_poly(:,i+j*p)= X(:,j+1).^i;
    end   
end

% =========================================================================

end
