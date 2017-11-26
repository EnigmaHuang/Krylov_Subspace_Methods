function [W, H, beta] = GMRES_Householder(A, r0, m)
% Householder for orthonormalizing Krylov subspace $K_{m}(A, r0)$
% W: the Householder unit vectors; H, beta: for GMRES
	n    = size(A, 1);
	beta = -norm(r0, 2);
	W    = zeros(n, m + 1);
	H    = zeros(m + 1, m);
	z    = r0 / -beta;
	
	for j = 1 : m + 1
		% Determine Pj (form Householder unit vector wj)
		w = z;
		w(1 : j-1) = 0;
		w(j) = w(j) + sign(w(j)) * norm(w(j : n), 2);
		w = w / norm(w, 2);
		W(:, j) = w;
		
		% h_{j-1} = Pj * z 
		if (j > 1)
			h = z - 2 * w * (w' * z);
			H(:, j - 1) = h(1 : m + 1);
		end
		
		% Form P1 * P2 * P3 ... Pj * ej.
		% v = Pj * ej = ej - 2 * w * w' * ej
		v = -2 * (w(j)') * w;
		v(j) = v(j) + 1;
		% v = P1 * P2 * ... Pjm1 * (Pj * ej)
		for k = (j-1) : -1 : 1
		    v = v - W(:, k) * (2 * (W(:, k)' * v));
		end
		% Explicitly normalize v to reduce the effects of round-off.
		v = v / norm(v, 2);
		% Form z = Pj * Pj-1 * ... P1 * A * v.
		v = A * v;
		for k = 1 : j
		    v = v - W(:, k) * (2 * (W(:, k)' * v));
		end
		z = v;
	end
end