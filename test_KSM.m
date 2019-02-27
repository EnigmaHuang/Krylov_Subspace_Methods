clear;
clf;

A = delsq(numgrid('S', 50));
b = rand(size(A, 1), 1);

cd CG
[~, ~, ic_cg, rn_cg] = CG_Classic(A, b);

cd ../BiCG
[~, ~, ic_bicgs, rn_bicgs] = BiCGStab(A, b);
[~, ~, ic_bicg, rn_bicg] = BiCG(A, b);

cd ../CR
[~, ~, ic_cr, rn_cr] = CR(A, b);

cd ../FOM
[~, ~, ic_fom, rn_fom] = FOM(A, b, 1e-9, 100, 50);

cd ../GMRES
[~, ~, ic_gmres, rn_gmres] = GMRES(A, b, 50, 1e-9, 100);

cd ..

semilogy(1 : ic_bicgs, rn_bicgs, 'r-'), hold on
semilogy(1 : ic_bicg,  rn_bicg,  'y-'), hold on
semilogy(1 : ic_cr,    rn_cr,    'g-'), hold on
semilogy(1 : ic_cg,    rn_cg,    'c-'), hold on
semilogy(1 : ic_fom,   rn_fom,   'b-'), hold on
semilogy(1 : ic_gmres, rn_gmres, 'm-'), hold on

xmin = 0;
xmax = max([ic_bicgs+1, ic_bicg+1, ic_cr+1, ic_cg+1, ic_fom+1, ic_gmres+1]) + 1;
ymin = min([min(rn_bicgs), min(rn_bicg), min(rn_cr), min(rn_cg), min(rn_fom), min(ic_gmres)]) * 0.8;
ymax = max([max(rn_bicgs), max(rn_bicg), max(rn_cr), max(rn_cg), max(rn_fom), max(ic_gmres)]) * 1.2;
axis([xmin xmax ymin ymax]);

xlabel('Iterations'), ylabel('Residual 2-norm'), grid on, hold on
legend('BiCGStab', 'BiCG', 'CR', 'CG-Classic', 'FOM-50', 'GMRES-50'), hold on
title_str1 = 'Krylov Subspace Methods for Solving Ax = b';
title_str2 = 'Matrix: delsq(numgrid(''S'', 50))';
title({title_str1; title_str2}), hold off