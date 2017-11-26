function [x, converged, iter_cnt, res_norm] = CR(A, b, res_tol, max_iter)
% Conjugate Residual algorithm
% Correspond to Algoithm 6.20 in Yousef Saad's "Iterative Methods for Sparse Linear System (2nd Edition)"
	if (nargin < 3)	res_tol  = 1e-9; end
	if (nargin < 4)	max_iter = 1000; end
	
	n = size(A, 1);
	x = zeros(n, 1);
	r = b - A * x;
	p = r;
	
	residual = norm(r, 2);
	res_stop = residual * res_tol;
	iter_cnt = 1;
	res_norm(iter_cnt) = residual;
	
	converged = 0;
	Ap = A * p;
	Ar = A * r;
	while ((iter_cnt < max_iter) && (residual > res_stop))
		alpha = r' * Ar / (Ap' * Ap);
		x = x + alpha * p;
		
		r_old  = r;
		Ar_old = Ar;
		r  = r - alpha * Ap;	
		Ar = A * r;
		
		beta = r' * Ar / (r_old' * Ar_old);
		p = r + beta * p;
		Ap = Ar + beta * Ap;
		
		iter_cnt = iter_cnt + 1;
		residual = norm(r, 2);
		res_norm(iter_cnt) = residual;
	end
	if (residual <= res_stop) converged = 1; end
end