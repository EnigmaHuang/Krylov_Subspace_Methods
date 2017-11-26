function [V, H, beta] = Arnoldi_MGS(A, r0, m)
% Modified Gram-Schmidt for orthonormalizing Krylov subspace $K_{m}(A, r0)$
% V: the orthonormalized basis; H, beta: for FOM
	beta = norm(r0, 2);
	n = size(A, 1);
	V = zeros(n, m + 1);
	H = zeros(m + 1, m);
	V(:, 1) = r0 / beta;
	for j = 1 : m
		w = A * V(:, j);
		for i = 1 : j
			H(i, j) = w' * V(:, i);
			w = w - H(i, j) * V(:, i);
		end
		H(j + 1, j) = norm(w, 2);
		if (H(j + 1, j) == 0) 
			break;
		end
		V(:, j + 1) = w / H(j + 1, j);
	end
end