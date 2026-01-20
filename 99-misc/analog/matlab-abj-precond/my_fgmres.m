function [x, resvec] = my_fgmres(A, b, tol, maxit, M, x0)

% Flexible GMRES (FGMRES) implementation. TODO: make more efficient and
% clean up code. This version is a very computationally inefficient version
% for quick prototyping and experimentation. May encounter some stability
% and robustness issues for very ill-conditioned problems.
%
% INPUTS
%   A      | Coefficient matrix of linear system
%   b      | Right-hand side vector
%   tol    | Solver tolerance (residual norm)
%   maxit  | Maximum number of iterations
%   M      | Preconditioner (matrix or function handle)
%   x0     | Initial guess
%
% OUTPUTS
%   x      | Solution vector
%   resvec | Vector of residual norms at each solver iteration
%
% Shikhar Shah (7 Nov 2025)
% sshah80@emory.edu

% SETUP AND ARGUMENT HANDLING

n = size(b, 1);
resvec = zeros(maxit+1, 1);

if isa(A, 'function_handle')
    Afun = A;
else
    Afun = @(x) A*x;
end

if isempty(M)
    Mfun = @(x) x;
elseif isa(M, 'function_handle')
    Mfun = M;
else
    Mfun = @(x) M\x;
end

if isempty(tol)
    tol = 1e-6;
end

if isempty(maxit)
    maxit = min(100, n);
end

if isempty(x0)
    x0 = zeros(size(b));
end

% ITERATION LOOP

r0 = b - Afun(x0);
beta = norm(r0);
resvec(1) = norm(Mfun(b)) ./ norm(b);
V = (1/beta) * r0;
for jx = 1:maxit
    Z(:, jx) = Mfun(V(:, jx));
    w = Afun(Z(:, jx));
    for ix = 1:jx
        H(ix, jx) = w' * V(:, ix);
        w = w - H(ix, jx)*V(:, ix);
    end
    H(jx+1, jx) = norm(w);
    V(:, jx+1) = (1 / H(jx+1, jx)) * w;

    e1 = zeros(jx+1, 1);
    e1(1) = 1;
    y = H(1:(jx+1), 1:jx) \ (beta*e1);
    x = x0 + Z(:, 1:jx)*y;
    resvec(jx+1) = norm(Mfun(b - Afun(x))) ./ norm(b);
    if resvec(jx+1) <= tol
        resvec = resvec(1:(jx+1));
        break
    end
end

end