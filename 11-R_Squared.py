# Programming Linear Regression
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# Very important to declare datatype, otherwise wrong answer
xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
ys = np.array([2, 4, 6, 8, 10, 12], dtype = np.float64)


m = (mean(xs)*mean(ys)-mean(xs*ys)) / (mean(xs)**2 - mean(xs*xs))
b = mean(ys)-m*mean(xs)
print (m, b)

prediction = [m*i+b for i in xs]

y_mean = mean(ys)
SE_ymean = sum([(y-y_mean)**2 for y in ys])
SE_yline = sum([(yactual-ypredict) for (yactual, ypredict) in zip(ys, prediction)])

print("RSquared = ", 1 - SE_yline/SE_ymean)
