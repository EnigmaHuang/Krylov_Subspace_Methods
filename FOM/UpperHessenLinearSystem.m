function [y, resvec] = UpperHessenLinearSystem(H, beta)
% Solve H(1 : m, 1 : m) * y = beta * e_1
% H is a m+1 row m column upper Hessenberg matrix
% H and beta are from Arnoldi algorithm
% resvec is the vector that contains the residual norm of each step
	m = size(H, 2);
	b = zeros(m, 1); b(1) = beta;
	resvec = zeros(m, 1);
	subdiag_H = diag(H, -1);
	
	% Use Gauss elimination to transform H(1 : m, 1 : m) into an upper triangular
	for i = 1 : m - 1
		k = -H(i + 1, i) / H(i, i);
		H(i + 1, i : end) = H(i + 1, i : end) + k * H(i, i : end);
		b(i + 1) = b(i + 1) + k * b(i);
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
	
	% Proposition 6.7 & formula (6.18)
	for j = 1 : m
		resvec(j) = subdiag_H(j) * abs(y(j));
	end
end