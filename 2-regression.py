import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# Goal : Predict the Adj. Close 0.01% datapoints later for Google stock.
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Close']*100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0

# Adj. Close will be the label, then we will shift it up.
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# len(df), number of values in dataframe df
# How far into the future do we want to predict, 10% here.
forecast_out = int(math.ceil(0.01*len(df)))
print("forecast_out", forecast_out)
df['label'] = df['Adj. Close'].shift(-forecast_out)
# some of the values will become NaN now, so we want to remove them out.
df.dropna(inplace=True)


X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# normalize
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 20)

rgr = LinearRegression()
rgr.fit(X_train, y_train)


# R^2 score, percentage of variance explained. Closer to 1, the better. Check link
# https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/r-squared-or-coefficient-of-determination
# How to interpret R^2 - http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit

accuracy = rgr.score(X_test, y_test)
print("Accuracy", accuracy)

prediction = rgr.predict(X_test)
print("RMSE", np.linalg.norm(y_test-prediction))
