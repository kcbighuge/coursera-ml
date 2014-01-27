function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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

%%% set up i x j matrix to test all possible permutations of C & sigma
chooseParam = [.1; .3; .6; 1];
errorMat = zeros(length(chooseParam), length(chooseParam));

%%% loop through all possible permutations of C & sigma
for i = 1:length(chooseParam)
	C = chooseParam(i);
	for j = 1:length(chooseParam)
		sigma = chooseParam(j);
		fprintf(['\nParameters are: C = %f and sigma = %f'], chooseParam(i), chooseParam(j));
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		pred = svmPredict(model, Xval);
		errorMat(i,j) = mean(double(pred ~= yval));
		errorMat
		min(min(errorMat))
	end
end

%%% find location of minimum in error matrix
[rowMin,colMin] = find(errorMat == min(min(errorMat)))

%%% assign row position to C, col position to sigma
C = chooseParam(rowMin(1));
sigma = chooseParam(colMin(1));


% =========================================================================

end
