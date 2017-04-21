function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a1 = [ones(m, 1) X]; 
z2 = Theta1 * a1';
a2 = [ones(1,m);sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

k = 1;
temp = 0;

for ind = 1:m
	typ = y(ind, 1);
	tempY = zeros(1, num_labels);
	tempY(typ) = 1;
	tempHTheta = a3(:, ind);
	temp = temp + sum( ( -tempY * log(tempHTheta) ) - ( (1 - tempY) * log(1 - tempHTheta) ) );
end


J = temp/m;

lr1 = 0;
lr2 = 0;

for row = 1:hidden_layer_size
	lr1 = lr1 + sum(Theta1(row, 2:end) .^ 2);
end
for row = 1:num_labels
	lr2 = lr2 + sum(Theta2(row, 2:end) .^ 2);
end

J = J + ( (lr1 + lr2) * lambda ) / (2 * m);




for ds = 1:m
	a_1 = [1 X(ds, :)]; 			
	z_2 = Theta1 * a_1'; 		
	a_2 = [1 ; sigmoid(z_2)]; 	
	z_3 = Theta2 * a_2; 			
	a_3 = sigmoid(z_3);			


	original_y = zeros(num_labels ,1); 
	original_y( y(ds) ) = 1;
	delta_3 = a_3 - original_y;	


	delta_2 = ( Theta2(:, 2 : end)' * delta_3 ) .* sigmoidGradient(z_2);


	Theta1_grad = Theta1_grad + (delta_2 * a_1);
	Theta2_grad = Theta2_grad + (delta_3 * a_2');
end
Theta1_grad(:, 1) = (Theta1_grad(:, 1) ./ m);
Theta1_grad(:, 2:end) = (Theta1_grad(:, 2:end) .+ (lambda .* Theta1(:, 2:end))) ./ m; 

Theta2_grad(:, 1) = (Theta2_grad(:, 1) ./ m);
Theta2_grad(:, 2:end) = (Theta2_grad(:, 2:end) .+ (lambda .* Theta2(:, 2:end))) ./ m; 

grad = [Theta1_grad(:) ; Theta2_grad(:)];






end
