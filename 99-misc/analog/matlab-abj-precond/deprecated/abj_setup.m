function P_info = abj_setup(A, num_blocks, rpu_settings)

% Set up array of structs for the analog block Jacobi preconditioner.
% Each cell contains the applicable index bounds [idx_s, idx_e] and an
% analog matrix containing inv(A(idx_s:idx_e, idx_s:idx_e)).
%
% INPUTS
%   A            | Coefficient matrix of linear system
%   num_blocks   | Number of diagonal blocks to use (Jacobi preconditioner
%                  uses num_blocks = n
%   rpu_settings | Settings struct for the RPU
%
% OUTPUTS
%   P_info       | Array of structs containing analog preconditioner
%
% Shikhar Shah (7 Nov 2025)
% sshah80@emory.edu

P_info = cell(num_blocks, 1);
n = size(A, 1);
block_size = ceil(n / num_blocks);
index_bounds = [1:block_size:n, n+1]';

for ix = 1:num_blocks
    P_info{ix}.idx_s = index_bounds(ix);
    P_info{ix}.idx_e = index_bounds(ix+1) - 1;
    P_info{ix}.rpu = RPU_Analog_Basic(rpu_settings);
    inv_diag_block = inv(A(P_info{ix}.idx_s:P_info{ix}.idx_e, ...
                           P_info{ix}.idx_s:P_info{ix}.idx_e));
    P_info{ix}.rpu.analog_write(inv_diag_block);
end

end