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
from keras.layers import Dense, LSTM, Dropout

import plotly.graph_objs as go
import plotly.offline as pyo

def model_prediction(df):
    Y=df.filter(["Close"])
    X=df.drop(['Close'],axis=1)

    Y['Date']=Y.index

    X=X.values
    Y=Y.values
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.05, random_state=10)

    Y_train_data=pd.DataFrame(Y_train[:,1])
    Y_val_data=pd.DataFrame(Y_val[:,1])

    Y_train=np.delete(Y_train,1,axis=1)
    Y_val=np.delete(Y_val,1,axis=1)


    scaler = MinMaxScaler(feature_range=(0,1))
    Y_train=scaler.fit_transform(Y_train)
    y_val=scaler.fit(Y_val)
#that is the bottom layer and it goes further and further. 
#return sequences being true means all of the hidden weights and baises will move to the next layer. 
#if it is set to false, then it won't and only the true output from that layer will move to the next layer. 
    model = Sequential()
#that is the bottom layer and it goes further and further. 
#return sequences being true means all of the hidden weights and baises will move to the next layer. 
#if it is set to false, then it won't and only the true output from that layer will move to the next layer. 
    model.add(LSTM(169, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(84, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(40))
    model.add(Dense(1)) 
    model.compile(optimizer='adam', loss='mean_squared_error')
#optimizer is how you train the data. usually you would use gradient descent to update the parameters. 
#adam is another option which is pretty much gradient descent based on the loss function but it's a bit better and works a bit differnetly. 
#and the loss is the function. 

# Train the model
    model.fit(X_train, Y_train, batch_size=5, epochs=5,shuffle=True)
    #so you have a trained model now. so you are now testing. 
    predictions = model.predict(X_val)

#the prediction for the model rn is now in scaled model. so you are rescaling it backwards
    predictions = scaler.inverse_transform(predictions)

    r2s=r2_score(Y_val, predictions)
#that is the r2 score. mulitplication of that value with 100 will give you the percentage value of accuracy. the closer this is to 1 is better. 
    rsme=np.sqrt(np.mean(((predictions - Y_val) ** 2)))
    todays_data=df.iloc[-1]
    todays_data.drop(['Close'],inplace=True)
    todays_data=np.array([todays_data])
    data_rn=model.predict(todays_data)
    todays_value=scaler.inverse_transform(data_rn)
    #plt.plot(Y_val)
    #plt.plot(predictions)

    # Save the chart to a temporary file
    #chart_file = "Stock-Price-Prediction-Model\Test_Flask\static\Image\output-1.jpg"
    #plt.savefig(chart_file)
    #train_dates = df.index[:-len(X_val)]
    val_dates = df.index[-len(X_val):]

    # plot the time series graph of the close price of the training data
    #trace1 = go.Scatter(x=train_dates, y=scaler.inverse_transform(Y_train)[:,0], mode="lines", name="Close Price of Training Data")

    # plot the actual close price of the testing data
    trace2 = go.Scatter(x = list(range(len(Y_val[:,0]))), y=Y_val[:,0], name="Actual Close Price of Testing Data")

    # plot the predicted price of the testing data
    trace3 = go.Scatter(x = list(range(len(Y_val[:,0]))), y=predictions[:,0], name="Predicted Close Price of Testing Data")

    #data = [trace1, trace2, trace3]
    data = [trace2, trace3]
    layout = dict(title='Stock Price Prediction', xaxis_title='Date', yaxis_title='Price',margin=dict(l=80, r=80, t=80, b=80),plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)") 
    fig = dict(data=data, layout=layout)

    graph_html = pyo.plot(fig, output_type="div")
    
    strout= "The predicted value is:"+str(todays_value[0][0])+" with accurary of:"+str((r2s*100))+" and RSME of:"+str(rsme)

    return strout,graph_html



app = Flask(__name__)  
 
@app.route('/', methods =["GET", "POST"])  
def gfg():  
    if request.method == "POST":
        stockname=request.form.get('sname')
        df=yk.download(tickers=stockname,period='90d',interval='60m')
        #go with either 60d and 30m or 3y and 1d or 60d and 60m or 90d and 60m
        if (((np.array(df.sum()))!=0).sum()==0):
            return render_template('index1.html', error="We are unable to fetch the data")
        va,graph_html=model_prediction(df)
        return render_template('graph.html',graph_html=graph_html)
    return render_template("index.html")  

  
if __name__ =="__main__":  
    app.run(debug = True)  