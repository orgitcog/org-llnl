clear
close all
rng(123)

%% PARAMETER SETUP

tol = 1e-10;
maxit = 400;
fig_lw = 2.0;
m = 50;
lap_shift = 0.0;

A = full(fd3d(m, m, 1, 0.1, 1.2, 0.3, 0.1)); % convection-diffusion problem
% A = full(fd3d(m, m, 1, 0, 0, 0, 0.0)); % Poisson problem

n = size(A, 1);
p = 5; % number of blocks for preconditioning

% Match HPEC 2023 paper parameters
rs = RPU_Analog_Basic.Get_Baseline_Settings;
rs.input_noise = 0.01;
rs.output_noise = 0.01;
rs.write_noise = 0.005;

b = randn(n, 1);
b = b ./ norm(b);
x0 = zeros(n, 1);

%% PRECONDITIONER SETUP

P1 = ComplexABJP(A, p, rs, 0, 1, 0.0, 0.0);
P2 = ComplexABJP(A, p, rs, 1, 0, 0.0, 0.2);
P3 = ComplexABJP(A, p, rs, 1, 1, 0.0, 0.2);
P4 = ComplexABJP(A, p, rs, 1, 0, 0.5i, 0.2);
P5 = ComplexABJP(A, p, rs, 1, 1, 0.5i, 0.2);
fprintf('Finished computing preconditioners...\n')

%% COMPUTE PRECONDITIONED SPECTRA AND RUN FGMRES

eig0 = eig(A);
[eig1, hull1] = P1.precond_eig(A);
[eig2, hull2] = P2.precond_eig(A);
[eig3, hull3] = P3.precond_eig(A);
[eig4, hull4] = P4.precond_eig(A);
[eig5, hull5] = P5.precond_eig(A);
fprintf('Finished computing preconditioned spectra...\n')

[xn, resvecn] = my_fgmres(A, b, tol, maxit, @(u) u, x0);
[x1, resvec1] = my_fgmres(A, b, tol, maxit, @(u) P1.apply(u), x0);
[x2, resvec2] = my_fgmres(A, b, tol, maxit, @(u) P2.apply(u), x0);
[x3, resvec3] = my_fgmres(A, b, tol, maxit, @(u) P3.apply(u), x0);
[x4, resvec4] = my_fgmres(A, b, tol, maxit, @(u) P4.apply(u), x0);
[x5, resvec5] = my_fgmres(A, b, tol, maxit, @(u) P5.apply(u), x0);
fprintf('Finished running FGMRES...\n')

%% VISUALIZATION

f1 = figure;
subplot(2, 3, 1)
plot(real(eig0), imag(eig0), '.k')
subplot(2, 3, 2)
plot(real(eig1), imag(eig1), '.r')
subplot(2, 3, 3)
plot(real(eig2), imag(eig2), '.g')
subplot(2, 3, 4)
plot(real(eig3), imag(eig3), '.c')
subplot(2, 3, 5)
plot(real(eig4), imag(eig4), '.b')
subplot(2, 3, 6)
plot(real(eig5), imag(eig5), '.m')

f2 = figure;
semilogy(resvecn, '--k', 'LineWidth', fig_lw)
hold on
semilogy(resvec1, '-r', 'LineWidth', fig_lw)
semilogy(resvec2, '-g', 'LineWidth', fig_lw)
semilogy(resvec3, '-c', 'LineWidth', fig_lw)
semilogy(resvec4, '-b', 'LineWidth', fig_lw)
semilogy(resvec5, '-m', 'LineWidth', fig_lw)
axis square
xlabel('Iteration number')
ylabel('Relative residual norm')
fontsize(14, "points")

% exportgraphics(f1, 'fgmres_complex_spectrum.pdf')
% exportgraphics(f2, 'fgmres_complex_converge.pdf')