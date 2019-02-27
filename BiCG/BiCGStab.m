function [x, converged, iter_cnt, res_norm] = BiCGStab(A, b, res_tol, max_iter)
% BiConjugate Gradients Stabilized method
% Correspond to Algoithm 7.7 in Yousef Saad's "Iterative Methods for Sparse Linear System (2nd Edition)"
	if (nargin < 3)	res_tol  = 1e-9; end
	if (nargin < 4)	max_iter = 1000; end
	
	n = size(A, 1);
	x = zeros(n, 1);
	r = b - A * x;
	p = r;
	Ap = A * p;
	r0s = r;
	rho = r' * r0s;
	
	residual = norm(r, 2);
	res_stop = residual * res_tol;
	iter_cnt = 1;
	res_norm(iter_cnt) = residual;
	
	converged = 0;
	while ((iter_cnt < max_iter) && (residual > res_stop))
		alpha = r' * r0s / (Ap' * r0s);
		s  = r - alpha * Ap;
		As = A * s;
		omega = (As' * s) / (As' * As);
		x = x + alpha * p + omega * s;

		rho_old = rho;
		r   = s - omega * As;
		rho = r' * r0s;
		
		beta = rho / rho_old * alpha / omega;
		p    = r + beta * (p - omega * Ap);
		Ap   = A * p;
		
		iter_cnt = iter_cnt + 1;
		residual = norm(r, 2);
		res_norm(iter_cnt) = residual;
	end
	if (residual <= res_stop) converged = 1; end
end