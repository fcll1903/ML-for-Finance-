import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization

########################################################################################################################
# Randomly generate Black-Scholes variables So,K,r,t,vol for computing call and put option prices;
# Build a ML model to predict prices of either call or option contingent claims with varying:
#       - Stock Price (So),
#       - Strike Price (K),
#       - Time to maturity (t),
#       - Risk-free rate of return (r),
#       - Price Volatility (vol),
########################################################################################################################

# Get d1 for the CDF generator, so that N(d1) can be interpreted as the probability of the stock price (So) to be
# higher than the strike price (k), allowing the contingent claim to be in-the-money in the case of a call option
# and out of the money the case of a put option
def getd1(df):
    return (np.log(df["So"] / df["K"]) + df["t"] * (df["r"] + 0.5 * df["vol"] ** 2)) / (df["vol"] * np.sqrt(df["t"]))


# Get d2 for the CDF generator, so that N(d2) can be interpreted as the probability of the stock price (So) to be
# lower than the strike price (k), allowing the contingent claim to be in-the-money in the case of a put option
# and out of the money the case of a call option
def getd2(df):
    return (np.log(df["So"] / df["K"]) + df["t"] * (df["r"] - 0.5 * df["vol"] ** 2)) / (df["vol"] * np.sqrt(df["t"]))


# Get the call option price derived from the Black Scholes Equation - since this equation is derived from the
# Geometric brownian motion, where returns are assumed to be log-normal distributed, the CDF used if from the normal
# distribution [P(X<x), where X ~ N(mu, var) and x = d1 or d2]
def getcallprice(df):
    return df["So"] * stats.norm.cdf(getd1(df)) - df["K"] * stats.norm.cdf(getd2(df)) * np.exp(-df["r"] * df["t"])


# Get the put option price derived from the Black Scholes Equation - since this equation is derived from the
# Geometric brownian motion, where returns are assumed to be log-normal distributed, the CDF used if from the normal
# distribution [P(X<x), where X ~ N(mu, var) and x = -d1 or -d2]
def getputprice(df):
    return df["K"] * stats.norm.cdf(-getd2(df)) * np.exp(-df["r"] * df["t"]) - df["So"] * stats.norm.cdf(-getd1(df))


# Randomly generate samples out of the BS equation
def datagenerator(samples):
    rng = np.arange(0, samples, 1)
    df = pd.DataFrame(index=rng, columns=["So", "K", "t", "r", "vol"])
    df["So"] = np.random.uniform(1, 1000, samples) # Stock pricing varying from 1 to 1000
    df["K"] = np.random.uniform(1, 500, samples)  # Strike pricing varying from 1 to 500
    df["t"] = np.random.uniform(1, 100, samples)  # Time to maturity varying from 1 to 100
    df["vol"] = np.random.uniform(0.001, 1, samples) # Price Volatility varying from 0.1% to 100%
    df["r"] = np.random.uniform(0.001, 0.1, samples) # Risk-free rate varying from 0.1% to 10%
    df["call"] = getcallprice(df)
    df["put"] = getputprice(df)
    return df


def datagenerator_static(samples):
    rng = np.arange(0, samples, 1)
    df = pd.DataFrame(index=rng, columns=["So", "K", "t", "r", "vol"])
    df["So"] = np.random.uniform(1, 200, samples)
    df["K"] = samples * [100]
    df["t"] = np.random.uniform(1, 20, samples)
    df["vol"] = samples * [0.5]
    df["r"] = samples * [0.05]
    df["call"] = getcallprice(df)
    df["put"] = getputprice(df)
    return df


# Plotting the prices as a function of stock price as well as the pdf of prices
def plotprice(df, type):
    sns.distplot(df[type], hist=True, kde=False,
                 bins=int(180 / 5), color='blue',
                 hist_kws={'edgecolor': 'black'})
    plt.show()
    plt.scatter(df["So"], df[type])
    plt.show()


# Generate normalize variables and output their max and min
def normalizaton(df):
    norm = (df - df.min()) / (df.max() - df.min())
    return norm, df.max(), df.min()


def renormalization(df, maxin, minin):
    return df * (maxin - minin) + maxin

# Understand the effect of the time to maturity in the call price x stock price with fixed strike price
## sudodata = datagenerator_static(10000)
## plotprice(sudodata, "call")

###########################################
#       Start the ML model
###########################################

# Data Generator
df = datagenerator(500000)

# Normalize Data
dfnorm, dfmax, dfmin = normalizaton(df)

# Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(dfnorm[["So", "K", "t", "vol", "r"]], dfnorm[["call", "put"]],
                                                    test_size=0.50, random_state=42)

# Feed-forward ANN with 2 hidden layers of 100 neurons each
# Adam optimizer
# Early stopping for overfitting prevention
model = Sequential()
model.add(Dense(100, input_dim=len(X_train.columns), init='uniform', activation='relu'))
model.add(BatchNormalization())
model.add(Dense(100, init='uniform', activation='relu'))
model.add(Dense(len(y_train.columns), init='uniform', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
ES = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)]
Trainmodel = model.fit(X_train, y_train, nb_epoch=500, callbacks=ES, batch_size=20000, validation_data=[X_test,y_test], verbose=1)

# Evaluate the model
_, train_acc = model.evaluate(X_train, y_train, verbose=1)
_, test_acc = model.evaluate(X_test, y_test, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# # Plot training history
plt.plot(Trainmodel.history['loss'], label='train')
plt.plot(Trainmodel.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make predictions
Yout = model.predict(X_test, batch_size=None, verbose=1, steps=None)
call = list(zip(*Yout))[0]
put = list(zip(*Yout))[1]

# Compute error
y_test = y_test.reset_index()
errorcall , errorput = ([abs(call[i] - y_test['call'][i]) for i in range(len(call))]) , ([abs(put[i] - y_test['put'][i]) for i in range(len(put))])
errorcallmean , errorputmean = np.average(errorcall) , np.average(errorput)
errorcalldev , errorputdev= np.std(errorcall) , np.std(errorput)
print('Mean Error Call %%: %.3f, Mean Error Put %%: %.3f' % (100*errorcallmean, 100*errorcalldev))
print('Std Dev Error Call %%: %.3f, Std Dev Error Put %%: %.3f' % (100*errorputmean, 100*errorputdev))

# Plot ANN output vs Actual output
print(type(call),type(y_test['call']))
plt.scatter(call, y_test['call'],s=0.05)
plt.title("Difference in Call Price")
plt.show()
