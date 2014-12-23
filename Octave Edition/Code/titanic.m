%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('train_oct.txt');
testData = load('test_oct.txt');

%Split 70% of data into training set. 30% of data into test set.
X = data(1:630, [2:end]); y = data(1:630, 1);
testX = data([631:end], [2:end]); testY = data([631:end], 1);

%X = data(:, [2:end]); y = data(:, 1);
test = testData(1:end,1:end);

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.


%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression. You neeed to complete the code in 
%  costFunction.m

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);
[tm,tn]=size(testX);
[tsm,tsn]=size(test);

% Add intercept term to x and X_test
X = [ones(m, 1) X];
testX =[ones(tm, 1) testX];
test=[ones(tsm,1) test];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%

p = 3;

%Do not poly "sex" feature
newX=[X(:,2), X(:,4) ,X(:,5)]
newTestX=[testX(:,2), testX(:,4),testX(:,5)]
newTest=[test(:,2), test(:,4),test(:,5)]

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(newX, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly, X(:,3)];                   % Add Ones and "Sex" back to X_poly

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(newTest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test,test(:,3)];         % Add Ones and "Sex" back to X_poly_test

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(newTestX, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val,testX(:,3)];           % Add Ones and "Sex" back to X_poly_val

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =========== Part 4: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%

lambda = 0.3;
[theta] = trainLogisticReg(X_poly, y, lambda);

figure(1);
[error_train, error_val] = ...
    LearningCurveForLR(X_poly, y, X_poly_val, testY, lambda);

plot(1:m, error_train, 1:m, error_val);

title(sprintf('Logistic Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 630 0 1])
legend('Train', 'Cross Validation')

fprintf('Logistic Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, testY);

close all;

plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability


%Compute accuracy on our test set
p = predict(theta, X_poly_val);
pv= predict(theta, X_poly_test);

%Save Prediction into the file.
csvwrite('testoutput.csv',pv)

fprintf('Train Accuracy: %f\n', mean(double(p == testY)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;
