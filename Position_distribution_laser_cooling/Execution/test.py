import numpy as np
import matplotlib.pyplot as plt
theta=[np.random.uniform()*2*np.pi for _ in range(1000)]
s=[np.sin(theta[i]) for i in range(1000)]
c=[np.cos(theta[i]) for i in range(1000)]

plt.hist(s,bins=100)
plt.hist(c,bins=100)
plt.show()