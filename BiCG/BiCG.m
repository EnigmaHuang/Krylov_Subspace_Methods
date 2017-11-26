function [x, converged, iter_cnt, res_norm] = BiCG(A, b, res_tol, max_iter)
% BiConjugate Gradients method
% Correspond to Algoithm 7.3 in Yousef Saad's "Iterative Methods for Sparse Linear System (2nd Edition)"
	if (nargin < 3)	res_tol  = 1e-9; end
	if (nargin < 4)	max_iter = 1000; end
	
	n = size(A, 1);
	x = zeros(n, 1);
	r = b - A * x;
	p = r;
	rs = r;
	ps = p;
	r2 = r' * rs;
	AT = A';
	Ap = A * p;
	ATps = AT * ps;
	
	residual = norm(r, 2);
	res_stop = residual * res_tol;
	iter_cnt = 1;
	res_norm(iter_cnt) = residual;
	
	converged = 0;
	while ((iter_cnt < max_iter) && (residual > res_stop))
		alpha = r2 / (Ap' * ps);
		x = x + alpha * p;
		r = r - alpha * Ap;
		rs = rs - alpha * ATps;
		
		r2_old = r2;
		r2 = r' * rs;
		beta = r2 / r2_old;
		
		p  = r  + beta * p;
		ps = rs + beta * ps;
		Ap = A * p;
		ATps = AT * ps;
		
		iter_cnt = iter_cnt + 1;
		residual = norm(r, 2);
		res_norm(iter_cnt) = residual;
	end
	if (residual <= res_stop) converged = 1; end
end