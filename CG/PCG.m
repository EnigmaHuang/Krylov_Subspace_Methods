function [x, converged, iter_cnt, res_norm] = PCG(A, b, res_tol, max_iter, M)
% (Left) Preconditioned Conjugate Gradient method
% Correspond to Algoithm 9.1 in Yousef Saad's "Iterative Methods for Sparse Linear System (2nd Edition)"
	n = size(A, 1);
	
	if (nargin < 3)	res_tol  = 1e-9; end
	if (nargin < 4)	max_iter = 1000; end
	if (nargin < 5) M = eye(n);      end
	
	x = zeros(n, 1);
	r = b - A * x;
	z = M \ r;        % Left preconditioning; MATLAB does not suggest saving inv(M)
	p = z;
	rho = r' * z;
	rn_stop = norm(r, 2) * res_tol;
	
	iter_cnt = 1;
	res_norm(iter_cnt) = norm(r);
	
	converged = 0;
	while ((iter_cnt < max_iter) && (res_norm(iter_cnt) > rn_stop))
		s     = A * p;
		alpha = rho / (p' * s);

		x = x + alpha * p;
		r = r - alpha * s;
		
		rho_0 = rho;
		z     = M \ r;
		rho   = r' * z;
		
		beta = rho / rho_0;
		p    = z + beta * p;
		
		iter_cnt = iter_cnt + 1;
		res_norm(iter_cnt) = norm(r, 2);
	end
	if (res_norm(iter_cnt) <= rn_stop) converged = 1; end
end