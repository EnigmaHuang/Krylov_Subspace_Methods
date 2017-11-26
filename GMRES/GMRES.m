function [x, converged, iter_cnt, res_norms] = GMRES(A, b, res_tol, max_iter, restart, use_HH)
% Generalized Minimum Residual Method with restarting 
% Correspond to Algorithm 6.9, 6.10, 6.11 in Yousef Saad's "Iterative Methods for Sparse Linear System (2nd Edition)"
	n = size(A, 1);
	
	if (nargin < 3)
		res_tol  = 1e-9;
	end
	if (nargin < 5) 
		restart  = min(n, 10); 
	end
	if (nargin < 4) 
		max_iter = min(floor(n / restart), 10); 
	end
	if (nargin < 6) 
		use_HH = 1;  % Use Householder by default
	end 
	
	x = zeros(n, 1);
	r = b - A * x;
	residual = norm(r);
	stop_res = residual * res_tol;
	out_iter = 0;
	iter_cnt = 1;
	res_norms = zeros(max_iter * restart + 1, 1);
	res_norms(iter_cnt) = residual;
	
	converged = 0;
	while ((out_iter < max_iter) && (residual > stop_res))
		if (use_HH == 1) 
			[W, H, beta] = GMRES_Householder(A, r, restart);
			[y, resvec]  = UpperHessenLeastSquare(H, beta); 
			z = zeros(n, 1);
			for j = restart : -1 : 1
				% z = Pj * (z + e_j * y(j))
				z(j) = z(j) + y(j);
				z = z - 2 * W(:, j) * (W(:, j)' * z);
			end
		else
			[V, H, beta] = Arnoldi_MGS(A, r, restart);
			[y, resvec]  = UpperHessenLeastSquare(H, beta); 
			z = V(:, 1 : restart) * y(1 : restart);
		end
		
		x = x + z;
		r = b - A * x;
		
		for j = 1 : restart
			iter_cnt = iter_cnt + 1;
			res_norms(iter_cnt) = resvec(j);
			residual = min(residual, resvec(j))
			if (residual < stop_res)
				break;
			end
		end
	end
	if (residual <= stop_res) converged = 1; end
	
	res_norms = res_norms(1 : iter_cnt);
end