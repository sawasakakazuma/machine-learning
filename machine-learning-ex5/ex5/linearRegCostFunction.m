function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

D = X * theta - y;

J = D'*D ./ (2*m) + lambda ./ (2*m) .* theta(2:end,:)'*theta(2:end,:);


theta_new = theta;
theta_new(1) = 0;

grad = X'*D ./m + lambda ./ m .* theta_new;


%grad(1) = D'*X(:,1) / m;
%sum(D)/m;
%grad(2) = D'*X(:,2) / m + lambda/m*theta(2);




% =========================================================================

grad = grad(:);

end
