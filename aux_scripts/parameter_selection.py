import numpy as np
import math
l = 3 # Number of layers
security_level = 192 # 0, 128, 192, 256
L = 2 * l# Multiplicative length of the circuit
s = 24 # Best according to them.
coeff_modulus = [s] * L
