
import numpy as np
import matplotlib.pyplot as plt

FILENAME = 'v2/perspective_data.csv'

with open(FILENAME, newline='') as f:
  data = np.genfromtxt(FILENAME, delimiter=',')[1:8]
  for pId,x1,y1,x2,y2 in data:
    print(pId)
    plt.plot(x1/510, -y1/255, 'ro', x2/510, -y2/255, 'bo')
  plt.xlim(0, 1)
  plt.ylim(-1, 0)
  plt.show()