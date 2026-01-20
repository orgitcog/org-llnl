import math
import numpy as np
from aihwkit.linalg import AnalogMatrix
from aihwkit.simulator.configs import FloatingPointRPUConfig

# RPU configuration that matches MATLAB simulator
def matlab_config(rpu_config):
    rpu_config.device.dw_min = 0.001 # MATLAB simulator, dw_min_mean = 0.001
    rpu_config.device.dw_min_dtod = 0.3 # MATLAB simulator, dw_min_dtod = 0.3
    rpu_config.device.w_max_dtod = 0.2 # MATLAB simulator, weight_range_dtod = 0.2
    rpu_config.device.w_min_dtod = 0.2 # MATLAB simulator, weight_range_dtod = 0.2
    rpu_config.forward.inp_noise = 0.01 # MATLAB simulator, input_noise  = 0.01
    rpu_config.forward.out_noise = 0.02 # MATLAB simulator, output_noise = 0.02
    rpu_config.forward.w_noise = 0.002 # MATLAB simulator, write_noise = 0.002
    rpu_config.mapping.max_input_size = 2**12 # 4096 (up to 16^3 or 64^2 FD mesh)
    rpu_config.mapping.max_output_size = 2**12 # 4096 (up to 16^3 or 64^2 FD mesh)
    return rpu_config

# RPU configuration that matches HPEC 2023 paper, used for SISC paper submission
def sisc_config(rpu_config):
    rpu_config.device.dw_min = 0.001 # MATLAB simulator, dw_min_mean = 0.001
    rpu_config.device.dw_min_dtod = 0.3 # MATLAB simulator, dw_min_dtod = 0.3
    rpu_config.device.w_max_dtod = 0.2 # MATLAB simulator, weight_range_dtod = 0.2
    rpu_config.device.w_min_dtod = 0.2 # MATLAB simulator, weight_range_dtod = 0.2
    rpu_config.forward.inp_noise = 0.01 # HPEC 2023 paper, input_noise  = 0.01
    rpu_config.forward.out_noise = 0.01 # HPEC 2023 paper, output_noise = 0.01
    rpu_config.forward.w_noise = 0.005 # HPEC 2023 paper, write_noise = 0.005
    rpu_config.mapping.max_input_size = 2**12 # 4096 (up to 16^3 or 64^2 FD mesh)
    rpu_config.mapping.max_output_size = 2**12 # 4096 (up to 16^3 or 64^2 FD mesh)
    return rpu_config

# RPU configuration that matches a single-precision digital matrix
def float_config():
    return FloatingPointRPUConfig()

# Preconditioner is a list of ComplexBlock objects, one for each diagonal block
class ComplexBlock:
    def __init__(self, rpu_r, scale_r, rpu_i, scale_i, iden_coeff, bounds):
        self.rpu_r = rpu_r
        self.scale_r = scale_r
        self.rpu_i = rpu_i
        self.scale_i = scale_i
        self.iden_coeff = iden_coeff
        self.bounds = bounds

class ComplexPreconditioner:
    def __init__(self, matrix, num_blocks, rpu_config, use_analog, use_spai, inner_z, outer_z):
        self.blocks = []
        n = matrix.shape[0]
        block_size = math.ceil(n / num_blocks)
        index_bounds = np.arange(0, n, block_size)
        index_bounds = np.append(index_bounds, n)

        for ix in range(num_blocks):
            idx_s, idx_e = index_bounds[ix], index_bounds[ix + 1]

            if use_analog:
                real_config = rpu_config
                if np.imag(inner_z) != 0:
                    imag_config = rpu_config
                else:
                    imag_config = float_config()
            else:
                real_config = float_config()
                imag_config = float_config()

            if use_spai:
                # TODO: SpAI algorithm
                inv_diag_block = None
            else:
                inv_diag_block = np.linalg.inv(matrix[idx_s:idx_e, idx_s:idx_e] + inner_z*np.eye(idx_e - idx_s))

            inv_real_block = np.real(inv_diag_block)
            inv_imag_block = np.imag(inv_diag_block)

            scale_r = np.max(np.abs(inv_real_block))
            inv_real_block = inv_real_block / scale_r
            if np.imag(inner_z) != 0:
                scale_i = np.max(np.abs(inv_imag_block))
            else:
                scale_i = 1.0
            inv_imag_block = inv_imag_block / scale_i

            self.blocks.append(
                ComplexBlock(AnalogMatrix(inv_real_block.astype("float32"), real_config, realistic=False), scale_r,
                             AnalogMatrix(inv_imag_block.astype("float32"), imag_config, realistic=False), scale_i,
                             outer_z, [idx_s, idx_e])
            )

        return

    def apply(self, in_vector):
        n = in_vector.shape[0]
        in_vector_r = np.real(in_vector).astype("float32")
        in_vector_i = np.imag(in_vector).astype("float32")
        out_vector = np.zeros(n, dtype="complex128")

        for block in self.blocks:
            idx_s, idx_e = block.bounds
            out_vector[idx_s:idx_e] +=    block.scale_r * (block.rpu_r.matvec(in_vector_r[idx_s:idx_e])).astype("float64")
            out_vector[idx_s:idx_e] -=    block.scale_i * (block.rpu_i.matvec(in_vector_i[idx_s:idx_e])).astype("float64")
            out_vector[idx_s:idx_e] += 1j*block.scale_r * (block.rpu_r.matvec(in_vector_i[idx_s:idx_e])).astype("float64")
            out_vector[idx_s:idx_e] += 1j*block.scale_i * (block.rpu_i.matvec(in_vector_r[idx_s:idx_e])).astype("float64")

        return out_vector

    def extract(self):
        # TODO: need some way to perform cheat_read() in the Python simulator
        P = None
        return P

    def precond_eig(self, matrix):
        # TODO: need some way to perform cheat_read() in the Python simulator
        eigenvalues = None
        eigen_hull = None
        return eigenvalues, eigen_hull