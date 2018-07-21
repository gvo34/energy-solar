import os
import io
import numpy as np

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

    plt.scatter(model.predict(X_train_scaled), model.predict(X_train_scaled) - y_train_scaled, c="blue", label="Training Data")
    plt.scatter(model.predict(X_test_scaled), model.predict(X_test_scaled) - y_test_scaled, c="red", label="Testing Data")
    plt.legend()
    plt.hlines(y=0, xmin=y_test_scaled.min(), xmax=y_test_scaled.max())
    plt.title("Residual Plot for history of ", history)
    plt.tight_layout()
    plt.savefig("static/images/LR_residual.png")
    print("DONE LINEAR")
    return jsonify('Done')



@app.route('/')
def upload_file():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
