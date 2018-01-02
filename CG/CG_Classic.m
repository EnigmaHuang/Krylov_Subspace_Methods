function [x, converged, iter_cnt, res_norm] = CG_Classic(A, b, res_tol, max_iter)
% Classic Conjugate Gradient method
% Correspond to Algorith 6.18 in Yousef Saad's "Iterative 
% Methods for Sparse Linear System (2nd Edition)"
% Or Algoithm 2.1 in Gear's paper "s-step Iterative Methods  
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

		x = x + alpha * p;
		r = r - alpha * s;
		
		r2_old = r2;
		r2 = r' * r;
		
		beta = r2 / r2_old;
		p = r + beta * p;
		
		iter_cnt = iter_cnt + 1;
		res_norm(iter_cnt) = norm(r, 2);
	end
	if (r2 <= r2_eps) converged = 1; end
end