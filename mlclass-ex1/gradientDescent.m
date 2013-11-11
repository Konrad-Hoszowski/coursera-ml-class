function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	sth1 = 0;
	sth2 = 0;	
	for j=1:m
		hx = X(j,:)*theta;
		sth1 += (hx - y(j))*X(j,1);
		sth2 += (hx - y(j))*X(j,2);
	end
	
	th1 = theta(1) - (alpha/m)*sth1;
	th2 = theta(2) - (alpha/m)*sth2;
	theta = [th1;th2];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

J_history

end
