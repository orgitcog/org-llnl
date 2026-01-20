function u = abj_apply(P_info, u)

% Apply the analog block Jacobi preconditioner (see abj_setup.m for
% construction of the preconditioner).
%
% INPUTS
%   P_info | Array of structs containing preconditioner information
%   u      | (Residual) vector to precondition
%
% OUTPUTS
%   u      | Vector after preconditioning (in-place)
%
% Shikhar Shah (7 Nov 2025)
% sshah80@emory.edu

for ix = 1:size(P_info, 1)
    idx_s = P_info{ix}.idx_s;
    idx_e = P_info{ix}.idx_e;
    u(idx_s:idx_e, :) = P_info{ix}.rpu.analog_MV(u(idx_s:idx_e, :));
end

end