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
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add bias term to the X data matrix
X = [ones(m, 1) X];

#
# For each training example
#
for i=1:m
  
  # Setup Y vector
  # Convert from 4 -> 0001000000
  Y = [zeros(num_labels,1)];
  Y(y(i),1) = 1;
  
  # Forward propegate through the network
  A1 = X(i,:);
  Z2 = (sum((Theta1.*A1)'));
  A2 = sigmoid(Z2);
 
  # Insert bias term
  A2 = [ones(1, 1) A2];
  
  Z3 = (sum((Theta2.*A2)'))';
  Hx = sigmoid(Z3);

  sumterm = ((-Y'*log(Hx)) - ((1-Y)'*log(1-Hx)));
 
  J = J + sum(sumterm);
end
#Now need to scale J
J = (1/m) .* J;
##########################################################
# Regularization term
# For each theta layer
# Sum for each element excluding bias term
Theta1sum = sum(sum(Theta1(:,2:end).^2));
Theta2sum = sum(sum(Theta2(:,2:end).^2));
regularization = (lambda/(2*m))*(Theta1sum + Theta2sum);
J = J + regularization;
##########################################################
#Gradient calculation using back propegation
for t=1:m
  # Forward propegate through the network
  a1 = X(t,:);
  z2 = Theta1 * a1';
  
  # Insert bias term
  a2 = [1; sigmoid(z2)];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  # Configure Y output
  Y = [zeros(num_labels,1)];
  Y(y(t),1) = 1;
  delta_3 = a3 - Y; 
  
  # Backprop
  # Note to self; Forgot to include the bias term
  delta_2 = Theta2'*delta_3.* [1; sigmoidGradient(z2)];
  delta_2 = delta_2(2:end);
  
  # Calculate cumulative grad
  Theta1_grad = Theta1_grad + delta_2 * a1;
  Theta2_grad = Theta2_grad + delta_3 * a2';
end

Theta1Reg = Theta1;
Theta2Reg = Theta2;
Theta1Reg(:,1)=0;
Theta2Reg(:,1)=0;

Theta1_grad = Theta1_grad/m + (lambda/m).*Theta1Reg;
Theta2_grad = Theta2_grad/m + (lambda/m).*Theta2Reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
