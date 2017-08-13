import pandas as pd
import quandl, datetime
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

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


X = np.array(df.drop(['label'], 1)) # the 1 here is for axis

# normalize
X = preprocessing.scale(X)
X_recent = X[-forecast_out:]
X = X[:-forecast_out]

# some of the values will become NaN now, so we want to remove them out.
df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 20)

rgr = LinearRegression()
rgr.fit(X_train, y_train)

# Save the classifier as a pickle file
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(rgr, f)

pickle_in = open('linearregression.pickle', 'rb')
rgr = pickle.load(pickle_in)

# R^2 score, percentage of variance explained. Closer to 1, the better. Check link
# https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/r-squared-or-coefficient-of-determination
# How to interpret R^2 - http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit

accuracy = rgr.score(X_test, y_test)
print("Accuracy", accuracy)

prediction = rgr.predict(X_test)
print("RMSE", np.linalg.norm(y_test-prediction))

# Forecasting
forecast_set = rgr.predict(X_recent)



# Plotting
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix+one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# This is what the above code does, creates a new column, and for every date beyond X_train, fills it with nan and then adds the forecasted value.
#             Adj. Close  HL_PCT  PCT_change  Adj. Volume  label    Forecast
# Date
# 2017-07-25         NaN     NaN         NaN          NaN    NaN  960.050246
# 2017-07-26         NaN     NaN         NaN          NaN    NaN  957.889158
# 2017-07-27         NaN     NaN         NaN          NaN    NaN  953.845175
# 2017-07-28         NaN     NaN         NaN          NaN    NaN  936.012266
# 2017-07-29         NaN     NaN         NaN          NaN    NaN  943.529047

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)   # bottom right corner
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
