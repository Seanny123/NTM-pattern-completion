import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tasks

inc_res, res = tasks.generate_predict_sequence(20, 10)

plt.imshow(np.array(res).T, interpolation='nearest')
plt.xlabel("time")
plt.ylabel("input")
plt.show()

