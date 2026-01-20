import math
import numpy as np
from aihwkit.linalg import AnalogMatrix

def matlab_config(rpu_config):
    rpu_config.device.dw_min = 0.001 # MATLAB simulator, dw_min_mean = 0.001
    rpu_config.device.dw_min_dtod = 0.3 # MATLAB simulator, dw_min_dtod = 0.3
    rpu_config.device.w_max_dtod = 0.2 # MATLAB simulator, weight_range_dtod = 0.2
    rpu_config.device.w_min_dtod = 0.2 # MATLAB simulator, weight_range_dtod = 0.2
    rpu_config.forward.inp_noise = 0.01 # MATLAB simulator, input_noise  = 0.01
    rpu_config.forward.out_noise = 0.02 # MATLAB simulator, output_noise = 0.02
    rpu_config.forward.w_noise = 0.002 # MATLAB simulator, write_noise = 0.002
    rpu_config.mapping.max_input_size = 2048
    rpu_config.mapping.max_output_size = 2048
    return rpu_config

class ABJBlock:
    def __init__(self, rpu, scale, bounds):
        self.rpu = rpu
        self.scale = scale
        self.bounds = bounds

class ABJPreconditioner:
    def __init__(self, matrix, num_blocks, rpu_config):
        self.abj_info = []
        n = matrix.shape[0]
        block_size = math.ceil(n / num_blocks)
        index_bounds = np.arange(0, n, block_size)
        index_bounds = np.append(index_bounds, n)

        for ix in range(num_blocks):
            idx_s, idx_e = index_bounds[ix], index_bounds[ix + 1]
            inv_diag_block = np.linalg.inv(matrix[idx_s:idx_e, idx_s:idx_e])
            # inv_diag_block[np.isclose(inv_diag_block, 0, atol=1e-2)] = 0
            scale = np.max(np.abs(inv_diag_block))
            inv_diag_block /= scale
            self.abj_info.append(
                ABJBlock(AnalogMatrix(inv_diag_block.astype("float32"), rpu_config, realistic=False), scale, [idx_s, idx_e])
            )

    def apply(self, in_vector):
        n = in_vector.shape[0]
        in_vector_32 = in_vector.astype("float32")
        out_vector = np.zeros(n, dtype="float64")

        for block in self.abj_info:
            idx_s, idx_e = block.bounds
            out_vector[idx_s:idx_e] = block.scale * (block.rpu.matvec(in_vector_32[idx_s:idx_e])).astype("float64")

        return out_vector