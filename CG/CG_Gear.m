function [x, converged, iter_cnt, res_norm] = CG_Gear(A, b, res_tol, max_iter)
% CG method with different formular for alpha, by Chronogonlas Gear, 1992
% Correspond to Algoithm 2.3 in Gear's paper "s-step Iterative Methods  
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
	
	w = A * r;
	s = w;
	alpha  = r2 / (r' * w);
	beta   = 0;
	
	converged = 0;
	while ((iter_cnt < max_iter) && (r2 > r2_eps))
		x = x + alpha * p;
		s = w + beta  * s; 
		r = r - alpha * s;
		w = A * r;
		
		r2_old = r2;
		r2 = r' * r;
		
		beta  = r2 / r2_old;
		alpha = r2 / ((w' * r) - beta / alpha * r2);
		p = r + beta * p;
		
		iter_cnt = iter_cnt + 1;
		res_norm(iter_cnt) = norm(r, 2);
	end
	if (r2 <= r2_eps) converged = 1; end
end