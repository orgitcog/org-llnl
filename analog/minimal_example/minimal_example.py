import numpy as np
from aihwkit.linalg import AnalogMatrix
from aihwkit.simulator.presets import ReRamSBPreset

def main():
    A = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    x = np.array([[0.7, 0.7], [-0.4, -0.4], [0.2, 0.2]])

    exact_mult = A @ x[:, 0] # [0.05; 0.20; 0.35]

    rpu_config = ReRamSBPreset() # preset configuration option
    P = AnalogMatrix(A.astype("float32"), rpu_config, realistic=False)

    # the multiplication requires the analog matrix and the vector to be single-precision floats,
    # but we can simply downcast to float32 and upcast the result to float64
    analog_mult1 = P.matvec(x[:, 0].astype("float32")).astype("float64")
    analog_mult2 = P.matvec(x[:, 1].astype("float32")).astype("float64")

    # matmat complains about a dimensionality issue that I do not fully understand yet
    # since we are simply performing (3x3) x (3x2) --> (3x2)
    # 
    # analog_matmat = P.matmat(x.astype("float32")).astype("float64")

    print("Exact:  ", exact_mult)
    print("Analog: ", analog_mult1)
    print("Analog: ", analog_mult2)

if __name__ == "__main__":
    main()
