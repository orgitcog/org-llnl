classdef ComplexABJP
    properties
        blocks
    end
    
    methods
        %% Constructor for block Jacobi preconditioner
        function obj = ComplexABJP(A, num_blocks, rpu_settings, bool_analog, bool_spai, inner_z, outer_z)
            obj.blocks = cell(num_blocks, 1);
            n = size(A, 1);
            block_size = ceil(n / num_blocks);
            index_bounds = [1:block_size:n, n+1]';
            
            for ix = 1:num_blocks
                obj.blocks{ix}.idx_s = index_bounds(ix);
                obj.blocks{ix}.idx_e = index_bounds(ix+1) - 1;
                idx_list = (index_bounds(ix)):(index_bounds(ix+1) - 1);
                obj.blocks{ix}.iden_coeff = outer_z;
            
                if bool_analog
                    obj.blocks{ix}.real_rpu = RPU_Analog_Basic(rpu_settings);
                    if imag(inner_z) ~= 0
                        obj.blocks{ix}.imag_rpu = RPU_Analog_Basic(rpu_settings);
                    else
                        obj.blocks{ix}.imag_rpu = RPU_Digital();
                    end
                else
                    obj.blocks{ix}.real_rpu = RPU_Digital();
                    obj.blocks{ix}.imag_rpu = RPU_Digital();
                end
            
                if bool_spai
                    % SpAI parameters taken from Vasileios's MATLAB code
                    nnzA = nnz(A);
                    nnzAp = 40*floor(nnzA/n); 
                    tol_spai = 0.05;
                    inv_diag_block = spai(A(idx_list, idx_list) + inner_z*eye(length(idx_list)), nnzAp, tol_spai, 1);
                else
                    inv_diag_block = inv(A(idx_list, idx_list) + inner_z*eye(length(idx_list)));
                end
            
                obj.blocks{ix}.real_rpu.analog_write(real(inv_diag_block));
                obj.blocks{ix}.imag_rpu.analog_write(imag(inv_diag_block));
            end
        end

        %% Apply block Jacobi preconditioner to u (in-place)
        function u = apply(obj, u)
            for ix = 1:size(obj.blocks, 1)
                idx_list = (obj.blocks{ix}.idx_s:obj.blocks{ix}.idx_e);
                real_u = real(u(idx_list, :));
                imag_u = imag(u(idx_list, :));
                
                real_v = obj.blocks{ix}.real_rpu.analog_MV(real_u) - obj.blocks{ix}.imag_rpu.analog_MV(imag_u);
                imag_v = obj.blocks{ix}.real_rpu.analog_MV(imag_u) + obj.blocks{ix}.imag_rpu.analog_MV(real_u);

                u(idx_list, :) = real_v + 1i*imag_v + obj.blocks{ix}.iden_coeff*u(idx_list, :);
            end
        end

        %% Extract block Jacobi preconditioner into (dense) matrix P
        function P = extract(obj)
            n = obj.blocks{end}.idx_e;
            P = zeros(n, n);
            for ix = 1:size(obj.blocks, 1)
                idx_list = (obj.blocks{ix}.idx_s:obj.blocks{ix}.idx_e);
                P(idx_list, idx_list) = obj.blocks{ix}.real_rpu.cheat_read() + 1i*obj.blocks{ix}.imag_rpu.cheat_read() + ...
                                        obj.blocks{ix}.iden_coeff*eye(length(idx_list));
            end
        end

        %% Get preconditioned spectrum and its convex hull
        function [p_eigval, p_eighull] = precond_eig(obj, A)
            M = obj.extract;
            [~, p_eigval] = eig(A*M);
            p_eigval = diag(p_eigval);
            p_eighull = conv_hull_eig(p_eigval);
        end
    end
end