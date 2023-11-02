import numpy as np
import time


a = np.random.random((2000, 2000, 3))

s = time.time()
for _ in range(100):
    b = a
print(f"{time.time() - s}")

s = time.time()
for _ in range(100):
    c = a.copy()
print(f"{time.time() - s}")

