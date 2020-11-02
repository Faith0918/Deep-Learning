import numpy as np
import matplotlib.pyplot as plt
import matplotlib

x = np.arange(0,6,0.1)
y = np.sin(x)

plt.plot(x,y)
plt.savefig("mygraph.png")
#not working for some reason
#plt.show()
