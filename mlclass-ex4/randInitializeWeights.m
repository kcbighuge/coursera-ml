function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the column row of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% for Theta1, L_out = hidden_layer_size=25, L_in = input_layer_size=400
% for Theta2, L_out = num_labels=10, L_in = hidden_layer_size=25

% ====================== YOUR CODE HERE ===================================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first row of W corresponds to the parameters for the bias units
%

epsilon_init = 0.12;
% epsilon init = sqrt(6) / sqrt(L_in + L_out)
% s_l = number units in layer L (without bias unit), where l is in Theta_l
% L_in = s_l, L_out = s_l+1

W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;


% =========================================================================

end
