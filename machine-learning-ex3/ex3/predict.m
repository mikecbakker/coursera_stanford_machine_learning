function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

#
# For each training example
# - Forward propegate through the network
# 
for i=1:m
  A1 = X(i,:);
  Z2 = (sum((Theta1.*A1)'));
  A2 = sigmoid(Z2);
 
  # Insert bias term
  A2 = [ones(1, 1) A2];
  
  Z3 = (sum((Theta2.*A2)'))';
  Hx = sigmoid(Z3);
 
  [val,ix] = max(Hx);
  p(i,1) = ix;
end



% =========================================================================


end
