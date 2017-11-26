function [x, converged, iter_cnt, res_norms] = FOM(A, b, res_tol, max_iter, restart)
% Full Orthogonalization Method with restarting 
% Correspond to Algorithm 6.4, 6.5 in Yousef Saad's "Iterative Methods for Sparse Linear System (2nd Edition)"
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
		[V, H, beta] = Arnoldi_MGS(A, r, restart);
		[y, resvec]  = UpperHessenLinearSystem(H, beta); 
		
		x = x + V(:, 1 : restart) * y;
		r = b - A * x;
		
		for j = 1 : restart
			iter_cnt = iter_cnt + 1;
			res_norms(iter_cnt) = resvec(j);
			residual = min(residual, resvec(j));
			if (residual < stop_res)
				break;
			end
		end
		out_iter = out_iter + 1;
	end
	if (residual <= stop_res) converged = 1; end
	
	res_norms = res_norms(1 : iter_cnt);
end