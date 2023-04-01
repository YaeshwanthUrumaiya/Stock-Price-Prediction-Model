from flask import *
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yk
# those are the basics. 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import r2_score
#to get the test train data split and then scale them. 

from keras.models import Sequential #sequential is a type of neutral network where the bottom layer feeds the layer infront. going in a sequential order
from keras.layers import Dense, LSTM

import plotly.graph_objs as go
import plotly.offline as pyo

def model_prediction(df):
    Y=df.filter(["Close"])
    X=df.drop(['Close'],axis=1)
    X=X.values
    Y=Y.values
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.05, random_state=10)
    scaler = MinMaxScaler(feature_range=(0,1))
    Y_train=scaler.fit_transform(Y_train)
    y_val=scaler.fit(Y_val)
    model = Sequential()
#that is the bottom layer and it goes further and further. 
#return sequences being true means all of the hidden weights and baises will move to the next layer. 
#if it is set to false, then it won't and only the true output from that layer will move to the next layer. 
    model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
#optimizer is how you train the data. usually you would use gradient descent to update the parameters. 
#adam is another option which is pretty much gradient descent based on the loss function but it's a bit better and works a bit differnetly. 
#and the loss is the function. 

# Train the model
    model.fit(X_train, Y_train, batch_size=1, epochs=1)
    #so you have a trained model now. so you are now testing. 
    predictions = model.predict(X_val)

#the prediction for the model rn is now in scaled model. so you are rescaling it backwards
    predictions = scaler.inverse_transform(predictions)
    plt.plot(Y_val)
    plt.plot(predictions)

    # Save the chart to a temporary file
    chart_file = "Stock-Price-Prediction-Model\Test_Flask\static\Image\output-1.jpg"
    plt.savefig(chart_file)
    return str(predictions[-1][0]),predictions



app = Flask(__name__)  
 
@app.route('/', methods =["GET", "POST"])  
def gfg():  
    if request.method == "POST":
       # getting input with name = fname in HTML form
       first_name = request.form.get("fname")
       # getting input with name = lname in HTML form
       last_name = request.form.get("lname")
       stockname=request.form.get('sname')

       df=yk.download(tickers=stockname,period='60d',interval='30m')
       va,ds=model_prediction(df)
       return render_template('index1.html', my_string=va,predictions=ds)
    data = [go.Scatter(x=[1, 2, 3], y=[4, 5, 6])]
    layout = go.Layout(title="My Plot")
    fig = go.Figure(data=data, layout=layout)
    graph_html = pyo.plot(fig, output_type="div")
    return render_template("index1.html", graph_html=graph_html)  

  
if __name__ =="__main__":  
    app.run(debug = True)  