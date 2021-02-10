function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_vec = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
s_vec = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

c_size = length(c_vec);
s_size = length(s_vec);

e = zeros(c_size, s_size);

for ci = 1:c_size
  for si = 1:s_size
    
    c = c_vec(ci);
    s = s_vec(si);
    
    model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
    pred = svmPredict(model, Xval);
    e(ci,si) = mean(double(pred ~= yval));
    
    fprintf('%d, %d\n',ci,si);
    
  endfor  
endfor

[temp c_t] = min(e);
[value s_t] = min(temp);

C = c_vec(c_t(s_t));
sigma = s_vec(s_t);

% =========================================================================

end
