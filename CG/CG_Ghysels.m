function [x, converged, iter_cnt, res_norm] = CG_Ghysels(A, b, res_tol, max_iter)
% CG method by Ghysels & Vanroose, 2012
% Correspond to Algoithm 3 in Ghysels's paper "Hiding global synchronization 
% latency in the preconditioned Conjugate Gradient algorithm"
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
	s = 0;
	z = 0;
	
	converged = 0;
	while ((iter_cnt < max_iter) && (r2 > r2_eps))
		q = A * w;
		if (iter_cnt == 1)
			beta  = 0;
			alpha = r2 / (w' * r);
		else
			beta  = r2 / r2_old;
			alpha = r2 / ((w' * r) - beta / alpha * r2);
		end
		
		s = w + beta * s;
		p = r + beta * p;
		z = q + beta * z;
		
		x = x + alpha * p;
		r = r - alpha * s;
		w = w - alpha * z;

		r2_old = r2;
		r2 = r' * r;
		
		iter_cnt = iter_cnt + 1;
		res_norm(iter_cnt) = norm(r, 2);
	end
	if (r2 <= r2_eps) converged = 1; end
end