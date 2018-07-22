import os
import io
import numpy as np
## Dependencies
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
from sklearn.preprocessing import StandardScaler

# import keras
# from keras.preprocessing import image
# from keras.preprocessing.image import img_to_array
# from keras.applications.xception import (
#     Xception, preprocess_input, decode_predictions)
# from keras import backend as K

from flask import Flask, request, redirect, url_for, jsonify, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Data'

model = None
graph = None



# persistence model
def model_persistence(x):
    return x


@app.route('/Autoregression/<history>')
def Autoregression(history):
    print("Autoregression with past lookup of ", history)
    past_lookup = int(history)
    ## Preprocessed time serie from Energy Solar consumption
    filename = "monthdata.csv"
    output_data = "static/data/output"
    filepath = os.path.join(output_data,filename)
    months = pd.read_csv(filepath,index_col=False, header=0)
    # convert into real dates, set as index for time series
    months['dates'] = pd.to_datetime(months["YYYYMM"], format="%Y%m")
    timeserie = months[['dates','Value']]
    timeserie = timeserie.set_index('dates')
    series = timeserie
 
    # split dataset
    X = series.values
    train, test = X[1:len(X)-past_lookup], X[len(X)-past_lookup:]
    print("train observations ", len(train), "test observations: ",len(test))
    # scale
   
    X_scaler = StandardScaler().fit(train.reshape(-1, 1))
    train_scaled = X_scaler.transform(train.reshape(-1, 1))
    test_scaled = X_scaler.transform(test.reshape(-1, 1))
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
    plt.title("Prediction Plot for history of " + history)
    plt.plot(test_scaled)
    plt.plot(predictions, color='red')
    # Save our graph 
    plt.tight_layout()
    plt.savefig("static/images/ARmodel.png")
    score = {"MSE": MSE}
    print(" done AR ", score)
    return jsonify(score)

@app.route('/ARHistory/<history>')
def ARHistory(history):
    print("Autoregression History with past lookup of ", history)
    past_lookup = int(history)
    ## Preprocessed time serie from Energy Solar consumption
    filename = "monthdata.csv"
    output_data = "static/data/output"
    filepath = os.path.join(output_data,filename)
    months = pd.read_csv(filepath,index_col=False, header=0)
    # convert into real dates, set as index for time series
    months['dates'] = pd.to_datetime(months["YYYYMM"], format="%Y%m")
    timeserie = months[['dates','Value']]
    timeserie = timeserie.set_index('dates')
    series = timeserie
    from statsmodels.tsa.ar_model import AR
    # split dataset
    X = series.values
    train, test = X[1:len(X)-past_lookup], X[len(X)-past_lookup:]
    print("train observations ", len(train), "test observations: ",len(test))
    # scale
    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler().fit(train.reshape(-1, 1))
    train_scaled = X_scaler.transform(train.reshape(-1, 1))
    test_scaled = X_scaler.transform(test.reshape(-1, 1))
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
    plt.title("Prediction Plot for history of " + history)
    plt.plot(test_scaled)
    plt.plot(predictions, color='red')
    # Save our graph 
    plt.tight_layout()
    plt.savefig("static/images/ARmodel_history.png")
    score = {"MSE": MSE}
    print(" done ARHistory ", score)
    return jsonify(score)


@app.route('/Linear/<history>')
def Linear(history):
    import pandas as pd
    import matplotlib.pyplot as plt
    import os


    print("doing Linear with ", history)

    history = int(history)
    ## read dataset
    filename = "X.csv"
    output_data = "static/data/output" 
    filepath = os.path.join(output_data,filename)
    X = pd.read_csv(filepath,index_col=False, header=0)
    filename = "y.csv"
    output_data = "static//data/output" 
    filepath = os.path.join(output_data,filename)
    y = pd.read_csv(filepath,index_col=False, names=["Value"])

    # overfitting treatment 
    X = X.drop(columns=["lag12", "peek12"])


    ## Split
    X_train, X_test = X[1:len(X)-history], X[len(X)-history:]
    y_train, y_test = y[1:len(X)-history], y[len(X)-history:] 
    
    ## Scale
    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)

    plt.figure()
    plt.scatter(model.predict(X_train_scaled), model.predict(X_train_scaled) - y_train_scaled, c="blue", label="Training Data")
    plt.scatter(model.predict(X_test_scaled), model.predict(X_test_scaled) - y_test_scaled, c="red", label="Testing Data")
    plt.legend()
    plt.hlines(y=0, xmin=y_test_scaled.min(), xmax=y_test_scaled.max())
    plt.title("Residual Plot for history of " + str(history))
    plt.tight_layout()
    plt.savefig("static/images/LR_residual.png")
    print("DONE LINEAR")

    
    predictions = model.predict(X_test_scaled)
    MSE = mean_squared_error(y_test_scaled, predictions)
    r2 = model.score(X_test_scaled, y_test_scaled)
    score_linear = {"r2": r2,"MSE": MSE}
    print(score_linear)
    return jsonify(score_linear)



@app.route('/')
def upload_file():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
