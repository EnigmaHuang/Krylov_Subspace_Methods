function [x, converged, iter_cnt, res_norm] = CG_Johnson(A, b, res_tol, max_iter)
% CG with different formular for beta, by Lennart Johnson, 1983 (?)
% Correspond to Algoithm 2.2 in Gear's paper "s-step Iterative Methods  
% for Symmetric Linear Systems"
	if (nargin < 3)	res_tol  = 1e-9; end
	if (nargin < 4)	max_iter = 1000; end
	
	n = size(A, 1);
	x = zeros(n, 1);
	r = b - A * x;
	p = r;
	
	r2     = r' * r;
	r2_eps = r2 * res_tol * res_tol;
	r2_old = r2;
	iter_cnt = 1;
	res_norm(iter_cnt) = norm(r);
	
	converged = 0;
	while ((iter_cnt < max_iter) && (r2 > r2_eps))
		s = A * p;
		alpha = r2 / (p' * s);
		
		beta  = (alpha * alpha * (s' * s) - r2) / r2;
		x = x + alpha * p;
		r = r - alpha * s;
		
		p = r + beta  * p;
		r2 = r' * r;
		
		iter_cnt = iter_cnt + 1;
		res_norm(iter_cnt) = norm(r, 2);
	end
	if (r2 <= r2_eps) converged = 1; end
end