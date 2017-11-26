function [y, resvec] = UpperHessenLeastSquare(H, beta)
% Finding y to minimize ||beta * e_1 - H * y||_2
% H is a m+1 row m column upper Hessenberg matrix
% H and beta are from Arnoldi algorithm
% resvec is the vector that contains the residual norm of each step
	m = size(H, 2);
	b = zeros(m + 1, 1); b(1) = beta;
	resvec = zeros(m, 1);
	
	% Use Givens rotation to eliminate subdiagonal of H
	for i = 1 : m
		h_i_i   = H(i, i);
		h_ip1_i = H(i + 1, i);
		denominator = sqrt(h_i_i * h_i_i + h_ip1_i * h_ip1_i);
		s = h_ip1_i / denominator;
		c = h_i_i   / denominator;
		G = [c s; -s c];
		
		% Eliminate H(i, i + 1)
		H(i : i + 1, :) = G * H(i : i + 1, :);
		b(i : i + 1)    = G * b(i : i + 1);
		
		resvec(i) = abs(b(i + 1));
	end
	
	% Use backward substitution to solve the upper triangular system 
	% H(1 : m, 1 : m) * y(1 : m) = b(1 : m)
	y = zeros(m, 1);
	for i = m : -1 : 1
		y(i) = b(i) / H(i, i);
		for j = 1 : m - 1
			b(j) = b(j) - H(j, i) * y(i);
		end
	end
end