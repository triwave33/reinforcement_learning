import matplotlib.pyplot as plt
import numpy as np

ax = plt.gca()

plt.xlim(0,5)
plt.ylim(0,5)
plt.xlabel('x')
plt.ylabel('y')

for i in range(5):
    for j in range(5):
        # rect
        rect = plt.Rectangle(xy =(i,j) , width=1, height=1, fill=False)
        ax.add_patch(rect)
        # diag
        diag = plt.Line2D(xdata=(i,i+1), ydata=(j,j+1),color='k')
        ax.add_line(diag)
        diag = plt.Line2D(xdata=(i,i+1), ydata=(j+1,j),color='k')
        ax.add_line(diag)
        # text
        plt.text(i+ 0.70, j+0.45, "%s, %s" % (str(i), str(j)))
        plt.text(i+ 0.4, j+0.8, "%s, %s" % (str(i), str(j)))
        plt.text(i+ 0.05, j+0.45, "%s, %s" % (str(i), str(j)))
        plt.text(i+ 0.4, j+0.1, "%s, %s" % (str(i), str(j)))

plt.show()

