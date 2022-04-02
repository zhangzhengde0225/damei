import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
f = np.sin(x) / x
print(x.shape, f.shape)
plt.plot(x, f)
plt.title('xx')
plt.show()
