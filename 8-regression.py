# Programming Linear Regression
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# Very important to declare datatype, otherwise wrong answer
xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
ys = np.array([3, 4, 7, 5, 18, 13], dtype = np.float64)


m = (mean(xs)*mean(ys)-mean(xs*ys)) / (mean(xs)**2 - mean(xs*xs))
b = mean(ys)-m*mean(xs)
print (m, b)

prediction = [m*i+b for i in xs]

plt.plot(xs, ys)
plt.plot(xs, prediction)
plt.show()
