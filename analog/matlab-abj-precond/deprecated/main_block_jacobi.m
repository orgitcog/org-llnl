clear
close all
rng(123)
addpath ..\matlab-analog-sim

%% PARAMETERS

m = 32; % m^2 will be the A matrix size (Laplacian)
d = 10; % A = Lap + X X^T + diag_shift*I where X has d columns
d_sparsity = 0.10;
num_blocks = [4; 16; 64];
diag_shift = 0;
tol = 1e-12;
maxit = 200;
rpu_settings = RPU_Analog_Basic.Get_Baseline_Settings; % Default settings
fprintf('Write  noise = %.3f\n', rpu_settings.write_noise)
fprintf('Read   noise = %.3f\n', rpu_settings.read_noise)
fprintf('Input  noise = %.3f\n', rpu_settings.input_noise)
fprintf('Output noise = %.3f\n', rpu_settings.output_noise)

%% SETUP PROBLEM AND PRECONDITIONER

L = full(delsq(numgrid('S', m+2)));
n = size(L, 1);
X = sprandn(n, d, d_sparsity);
A = L + X*X' + diag_shift*eye(n);
b = randn(n, 1);
b = b ./ norm(b);

eigval_L = eig(L); % for debugging purposes
eigval_A = eig(A); % for debugging purposes
% spy(A)

P_info3 = abj_setup(A, num_blocks(1), rpu_settings);
P3 = @(u) abj_apply(P_info3, u);

P_info4 = abj_setup(A, num_blocks(2), rpu_settings);
P4 = @(u) abj_apply(P_info4, u);

P_info5 = abj_setup(A, num_blocks(3), rpu_settings);
P5 = @(u) abj_apply(P_info5, u);

%% FGMRES

[x1, ~, ~, ~, resvec1] = gmres(A, b, [], tol, maxit, @(u) u, [], zeros(n, 1)); % GMRES (no precond.)
[x2, ~, ~, ~, resvec2] = gmres(A, b, [], tol, maxit, @(u) diag(diag(A))\u, [], zeros(n, 1)); % GMRES (Jacobi)
[x3, resvec3] = my_fgmres(A, b, tol, maxit, P3, zeros(n, 1)); % FGMRES (ABJ num_blocks(1))
[x4, resvec4] = my_fgmres(A, b, tol, maxit, P4, zeros(n, 1)); % FGMRES (ABJ num_blocks(2))
[x5, resvec5] = my_fgmres(A, b, tol, maxit, P5, zeros(n, 1)); % FGMRES (ABJ num_blocks(3))

%% VISUALIZATION

f = figure;
semilogy(0:length(resvec1)-1, resvec1, '-k', 'LineWidth', 1.2)
hold on
semilogy(0:length(resvec2)-1, resvec2, '--k', 'LineWidth', 1.2)
semilogy(0:length(resvec3)-1, resvec3, '-r', 'LineWidth', 1.2)
semilogy(0:length(resvec4)-1, resvec4, '-g', 'LineWidth', 1.2)
semilogy(0:length(resvec5)-1, resvec5, '-b', 'LineWidth', 1.2)
xlabel('Iteration number')
ylabel('Relative residual norm')
title(sprintf('A = Lap + XX^T + zI, n = %i, d = %i, z = %.1f', n, d, diag_shift))
legend('GMRES (iden.)', 'GMRES (Jacobi)', sprintf('FGMRES (ABJ%i)', num_blocks(1)), ...
       sprintf('FGMRES (ABJ%i)', num_blocks(2)), sprintf('FGMRES (ABJ%i)', num_blocks(3)), 'Location', 'ne')
axis square
fontsize(14, "points")
exportgraphics(f, 'fgmres_abj_comparison.pdf')