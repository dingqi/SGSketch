function [res] = dist_hamming(X,Y)
%DIST_MINMAX Summary of this function goes here
%   Detailed explanation goes here

% X = embs(1,:);
% Y = embs;

res = sum((X-Y)~=0,2)./sum(Y~=0,2);

end

