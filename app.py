## Dependencies
import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
## Models
from statsmodels.tsa.ar_model import AR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from flask import Flask, request, redirect, url_for, jsonify, render_template

app = Flask(__name__)

## common routines
def read_dataset():
    filename = "X.csv"
    output_data = "static/data/output" 
    filepath = os.path.join(output_data,filename)
    X = pd.read_csv(filepath,index_col=False, header=0)
    filename = "y.csv"
    output_data = "static//data/output" 
    filepath = os.path.join(output_data,filename)
    y = pd.read_csv(filepath,index_col=False, names=["Value"])
    print("X ", X.columns)
    return X, y

def read_timeserie():
    ## Preprocessed time serie from Energy Solar consumption
    filename = "monthdata.csv"
    output_data = "static/data/output"
    filepath = os.path.join(output_data,filename)
    months = pd.read_csv(filepath,index_col=False, header=0)
    # convert into real dates, set as index for time series
    months['dates'] = pd.to_datetime(months["YYYYMM"], format="%Y%m")
    timeserie = months[['dates','Value']]
    timeserie = timeserie.set_index('dates')
    return timeserie    

def split_scale(series, past_look):
    X = series.values
    #split
    train, test = X[1:len(X)-past_look], X[len(X)-past_look:]
    print("train observations ", len(train), "test observations: ",len(test))
    # scale
    X_scaler = StandardScaler().fit(train.reshape(-1, 1))
    train_scaled = X_scaler.transform(train.reshape(-1, 1))
    test_scaled = X_scaler.transform(test.reshape(-1, 1))
    return train_scaled, test_scaled

def split_scale_Xy(X, y, past_look):
    ## Split
    X_train, X_test = X[1:len(X)-past_look], X[len(X)-past_look:]
    y_train, y_test = y[1:len(X)-past_look], y[len(X)-past_look:] 
    ## Scale
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    print(len(X_train), len(X_test), len(y_train), len(y_test))
    print("train observations ", len(X_train_scaled), "test observations: ",len(X_test_scaled))
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled


# persistence model
def model_persistence(x):
    return x


@app.route('/Autoregression/<history>')
def Autoregression(history):
    print("Autoregression with past lookup of ", history)
    past_lookup = int(history)
    
    series = read_timeserie()
 
    # split and scale dataset
    train_scaled, test_scaled = split_scale(series, past_lookup)
    
    # train autoregression
    model = AR(train_scaled)
    model_fit = model.fit()
    print('Lag: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    predictions = model_fit.predict(start=len(train_scaled), end=len(train_scaled)+len(test_scaled)-1, dynamic=False)
    for i in range(len(predictions)):
        print('predicted=%f, expected=%f' % (predictions[i], test_scaled[i]))
    MSE = mean_squared_error(test_scaled, predictions)
    # plot results
    plt.figure()
    
    plt.plot(test_scaled)
    plt.plot(predictions, color='red')
    plt.legend(['Solar Energy','Predictions'], loc='best')
    plt.title("Autoregression plot of " + history + " months")
    
    # Save our graph 
    plt.tight_layout()
    plt.savefig("static/images/ARmodel" + history+ ".png")
    score = {"MSE": MSE}
    print(" done AR ", score)
    return jsonify(score)

@app.route('/ARHistory/<past>')
def ARHistory(past):
    print("Autoregression History with past lookup of ", past)
    past_lookup = int(past)
    
    series = read_timeserie()

    # split and scale dataset
    train_scaled, test_scaled = split_scale(series, past_lookup)

    # train autoregression
    model = AR(train_scaled)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    print(f"AR model params window {window}, coef {coef}")
    # walk forward over time steps in test
    history = train_scaled[len(train_scaled)-window:]
    print("History for re-training: ",len(history))
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test_scaled)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test_scaled[t]
        predictions.append(yhat)
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    MSE = mean_squared_error(test_scaled, predictions)
    print('Test MSE: %.3f' % MSE)
   # plot results
    plt.figure()
    plt.title("Autoregression with retraining plot of " + past + " months")
    plt.plot(test_scaled)
    plt.plot(predictions, color='red')
    plt.legend(['Solar Energy','Predictions'], loc='best')
    # Save our graph 
    plt.tight_layout()
    plt.savefig("static/images/ARmodel_history" + past+ ".png")
    score = {"MSE": MSE}
    print(" done ARHistory ", score)
    return jsonify(score)


@app.route('/Linear/<history>')
def Linear(history):
    print("doing Linear with ", history)

    months = int(history)
    X, y = read_dataset()

    # overfitting treatment 
    X = X.drop(columns=["lag12", "peek12"])

    # split and train X and y
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = split_scale_Xy(X, y, months)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)

    predictions = model.predict(X_test_scaled)
    MSE = mean_squared_error(y_test_scaled, predictions)
    r2 = model.score(X_test_scaled, y_test_scaled)
    score_linear = {"r2": r2,"MSE": MSE}
    plt.figure()
    # plot
    plt.plot(y_test_scaled)
    plt.plot(predictions, color='red')
    plt.legend(['Solar Energy','Predictions'], loc='best')
    # Save our graph 
    plt.title("Linear Regression of " + history + " months")
    plt.tight_layout()
    plt.savefig("static/images/LinearRmodel" + history+ ".png")
    print("DONE LINEAR")
    print(score_linear)
    return jsonify(score_linear)


@app.route('/MLP/<history>')
def MLP(history):
    print("doing MLP with ", history)

    months = int(history)
    X, y = read_dataset()
    # overfitting treatment 
    X = X.drop(columns=["lag12", "peek12"])
    
    # split and train X and y
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = split_scale_Xy(X, y, months)
    
    mlp = MLPRegressor(max_iter=1000, learning_rate_init=0.1, random_state=0, learning_rate='adaptive',
                    activation='relu', solver='adam', tol=0.0, verbose=2 , hidden_layer_sizes = (20,20))

    y_train_ravel = np.ravel(y_train_scaled)
    mlp.fit(X_train_scaled, y_train_ravel)
    predictions = mlp.predict(X_test_scaled)
    MSE = mean_squared_error(y_test_scaled, predictions)
    r2 = mlp.score(X_test_scaled, y_test_scaled)

    # plot
    plt.figure()
    y_test_ravel = np.ravel(y_test_scaled)
    plt.plot(y_test_ravel)
    plt.plot(predictions, color='red')
    plt.legend(['Solar Energy','Predictions'], loc='best')
    plt.title("MLP of " + history + " months")
    # Save our graph 
    plt.tight_layout()
    plt.savefig("static/images/MLPmodel"+ history +".png")

    print(f"MSE: {MSE}, r2: {r2}")
    score = {"r2": r2,"MSE": MSE}
    return jsonify(score)


@app.route('/RF/<history>')
def RandomForrest(history):
    print("doing Random Forrest with ", history)

    months = int(history)
    X, y = read_dataset()

    # overfitting treatment 
    X = X.drop(columns=["lag12", "peek12"])
    
    # split and train X and y
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = split_scale_Xy(X, y, months)

    # results from RandomForrest RandomizedSearchCV call    
    rf = RandomForestRegressor(bootstrap=False,max_depth=None,max_features="sqrt",min_samples_leaf=1,min_samples_split=2,n_estimators=400)
    
    y_train_ravel = np.ravel(y_train_scaled)

    rf.fit(X_train_scaled, y_train_ravel)

    predictions = rf.predict(X_test_scaled)
    y_test_ravel = np.ravel(y_test_scaled)
    MSE = mean_squared_error(y_test_ravel, predictions)
    r2 = rf.score(X_test_scaled, y_test_ravel)

    plt.figure()
    plt.plot(y_test_ravel, color='blue')
    plt.plot(predictions, color='red')
    plt.legend(['Solar Energy','Predictions'], loc='best')
    plt.title("Random Forrest of " + history + " months")
    # Save our graph 
    plt.tight_layout()
    plt.savefig("static/images/RFmodel"+ history +".png")

    print(f"MSE: {MSE}, r2: {r2}")
    score = {"r2": r2,"MSE": MSE}
    return jsonify(score)



@app.route('/')
def upload_file():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
