#Prathamesh Sai
#Predicting Stock Market Prices Using Machine Learning
import quandl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

#we import train test split so that it can split our test data into the training portion and the testing portion.
from sklearn.model_selection import train_test_split

#we import preprocessing to standardize the data.
from sklearn import preprocessing

#we import LinearRegression as we will be using a Linear Regression model.
from sklearn.linear_model import LinearRegression

#use my API Key for Quandl.
quandl.ApiConfig.api_key = 'ygW4wsDpysSHgnNszi7b'

#get our dataframe, passing in which service we are using, and which stock we want to use.
#Gets over 20 years of data to work with.
dataframe = quandl.get("WIKI/NKE")

#We want to work with "Adj. Close" column, as it takes inflation into account.
#Adjusted close is the closing price after adjustments for all applicable splits and dividend distributions.
dataframe = dataframe[['Adj. Close']]

#uncomment to see the Adjusted Close prices
#print(dataframe)

#plot the adjusted close prices from 1997 to 2018.
dataframe['Adj. Close'].plot(figsize=(15,6), color='g')
plt.legend(loc='upper left')

#uncomment 2 lines below to see the plot.
#plt.xlim(xmax=datetime.date(2018,4,11))
#plt.show()

#Create a new variable to predict the stock price an integer number of days from now.
forecast = 30 #30 days from now
#Add a new column to our dataframe and shifting the data "forecast" units up
dataframe['Prediction'] = dataframe[['Adj. Close']].shift(-forecast)

#uncomment below to see the data.
#print(dataframe)

#The X variable will be the adjusted close prices
#Specify "1" at the end so that it will drop the column, and not the index.
X = np.array(dataframe.drop(['Prediction'], 1))

#Now, we standardize our data.
#This is because all the data has different ranges, different means, different standard deviations.
#By standardizing our data, we ensure that our dataset has 0 mean, and a standard deviation of 1.
X = preprocessing.scale(X)

#uncomment 4 lines below to see the the mean and the standard deviation
#print("The Mean of X is: ") #Will give a value close to 0, but not exactly 0.
#print(X.mean())
#print("The Standard Deviation of X is: ") #Will give a value of exactly 1.
#print(X.std())

#Make forecast_X equal to the last "forecast" days from the dataset
forecast_X = X[-forecast:]

#X will be equal to all but the last "forecast" from X.
X = X[:-forecast]

#Y will be equal to the column "prediction"
y = np.array(dataframe['Prediction'])

#Y will be equal to all but the last "forecast".
y = y[:-forecast]

#uncomment below to see how linear regression is working
#plt.plot(X, y)

#Now, X and y have the exact same length or column size for the features and labels training sets.

#After that, we must create training and testing data.
#We are going to split our data into 20% testing data, and 80% training data.
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

#create an estimator instance which is a classifier.
estimatorInstance = LinearRegression()

#Initialize our model
estimatorInstance.fit(train_X, train_y)

#Now that our data is fitted, we need to calculate a score.
#We get the confidence score of our data from scoring our testing data.
confidenceScore = estimatorInstance.score(test_X, test_y)

#uncomment below to see the confidence/score of our model (our confidence value should be close to 1)
#print("The confidence is: ")
#print(confidenceScore)

#now that we have confidence in our model, we can finally predict the stock market prices.
predicted_forecast = estimatorInstance.predict(forecast_X)

#uncomment to see the predicted values of the last 30 days.
#Each value corresponds to what our machine learning model thinks the price will be in 30 days from the day it was given.
#print(predicted_forecast)

#Now we will create an array to store 30 days.
dates = pd.date_range(start="2018-03-28", end="2018-04-26")

#Now we have to plot 2 line charts.

#1.plot our dates with our predicted forecast
plt.plot(dates, predicted_forecast, color='y')

#2.plot our adjusted close prices
dataframe['Adj. Close'].plot(color='g')

#zoom into the plot to see the predicted stock values.
plt.xlim(xmin=datetime.date(2017,4,26), xmax=datetime.date(2018,4,26))

#show the plot.
plt.show()