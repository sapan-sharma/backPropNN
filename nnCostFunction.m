function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%

X = [ones(m, 1) X];
y_matrix = eye(num_labels)(y,:);

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = X;
a2 = sigmoid(X * Theta1');  
a3 = sigmoid([ones(m, 1) a2] * Theta2');

Ju = (-1/m) * sum(sum(y_matrix  .*  log(a3)  +  (ones(size(y_matrix, 1),size(y_matrix, 2)) - y_matrix)  .*  log(ones(size(a3, 1),size(a3, 2)) - a3)));

Theta1_r = Theta1;
Theta1_r(:,1) = zeros( size(Theta1,1),1 );

Theta2_r = Theta2;
Theta2_r(:,1) = zeros( size(Theta2,1),1 );

Jr =  (lambda/(2*m)) * ( sum(sum(Theta1_r.^2) ) + sum(sum(Theta2_r.^2 )) );

J = Ju+Jr;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


a1 = X;
z2 =  a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3=  a2 * Theta2';
a3 = sigmoid(z3);


d3 = a3 - y_matrix;

% calc z2

d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

%disp("d2");
%disp(d2);

%disp("d3");
%disp(d3);

del1 = d2' * a1;
%disp(del1);
del2 = d3' * a2;                                                                  
%disp("del2");
%disp(del2);



%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



Theta1_grad = (1/m) * del1;

Theta2_grad = (1/m) * del2;

Theta1(:,1) = 0;
Theta1_grad = Theta1_grad + (lambda/m)*Theta1;

Theta2(:,1) = 0;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
