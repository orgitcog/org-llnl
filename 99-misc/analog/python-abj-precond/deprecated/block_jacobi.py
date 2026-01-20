import math
import numpy as np

class JacobiBlock:
    def __init__(self, matrix, bounds):
        self.matrix = matrix
        self.bounds = bounds

class BlockJacobiPreconditioner:
    def __init__(self, matrix, num_blocks):
        self.bj_info = []
        n = matrix.shape[0]
        block_size = math.ceil(n / num_blocks)
        index_bounds = np.arange(0, n, block_size)
        index_bounds = np.append(index_bounds, n)

        for ix in range(num_blocks):
            idx_s, idx_e = index_bounds[ix], index_bounds[ix + 1]
            inv_diag_block = np.linalg.inv(matrix[idx_s:idx_e, idx_s:idx_e])
            # inv_diag_block[np.isclose(inv_diag_block, 0, atol=1e-2)] = 0
            self.bj_info.append(JacobiBlock(inv_diag_block, [idx_s, idx_e]))

    def apply(self, in_vector):
        n = in_vector.shape[0]
        out_vector = np.zeros(n)

        for block in self.bj_info:
            idx_s, idx_e = block.bounds
            out_vector[idx_s:idx_e] = block.matrix @ in_vector[idx_s:idx_e]

        return out_vector