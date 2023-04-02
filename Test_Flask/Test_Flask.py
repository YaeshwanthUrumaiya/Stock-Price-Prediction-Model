from flask import *
import numpy as np 
import pandas as pd
import yfinance as yk


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 

from keras.models import Sequential
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

    Y_train=np.delete(Y_train,1,axis=1)
    Y_val=np.delete(Y_val,1,axis=1)


    scaler = MinMaxScaler(feature_range=(0,1))
    Y_train=scaler.fit_transform(Y_train)

    model1 = Sequential()
 
    model1.add(LSTM(169, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    model1.add(Dropout(0.2))
    model1.add(LSTM(84, return_sequences=True))
    model1.add(Dropout(0.2))
    model1.add(LSTM(64, return_sequences=False))
    model1.add(Dropout(0.2))
    model1.add(Dense(40))
    model1.add(Dense(1)) 

    model1.compile(optimizer='adam', loss='mean_absolute_error')

    model1.fit(X_train, Y_train, batch_size=5, epochs=5,shuffle=True)

    predictions = model1.predict(X_val)

    predictions = scaler.inverse_transform(predictions)
    rsme=np.sqrt(np.mean(((predictions - Y_val) ** 2)))

    todays_data=df.iloc[-1]
    todays_data.drop(['Close'],inplace=True)
    todays_data=np.array([todays_data])
    data_rn=model1.predict(todays_data)
    todays_value=scaler.inverse_transform(data_rn)

    t1 = go.Scatter(x = list(range(len(Y_val[:,0]))), y=Y_val[:,0], name="Actual Close Price of Testing Data")
    t2 = go.Scatter(x = list(range(len(Y_val[:,0]))), y=predictions[:,0], name="Predicted Close Price of Testing Data")

    data = [t1, t2]
    layout = dict(title='Stock Price Prediction', xaxis_title='Date', yaxis_title='Price',margin=dict(l=100, r=100, t=200, b=200),plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)") 
    fig = dict(data=data, layout=layout)

    graph1 = pyo.plot(fig, output_type="div")
    
    s1=str(round(todays_value[0][0], 2))
    s2=str(np.ceil(rsme))

    return s1,graph1,s2



app = Flask(__name__)  
 
@app.route('/', methods =["GET", "POST"])  
def test():  
    if request.method == "POST":

        stockname=request.form.get('sname')
        per=request.form.get('per')

        inter=None

        if(per=='60d'):
            inter='30m'

        if(per=='30d'):
            inter='5m'

        elif (per=='90d'):
            inter='60m'

        elif(per=="3y"):
            inter='1d'

        else:
            pass  

        try:
            df=yk.download(tickers=stockname,period=per,interval=inter)
        except:
            return render_template("index.html", error="We are unable to fetch the data")

        if (((np.array(df.sum()))!=0).sum()==0):
            return render_template("index.html", error="We are unable to fetch the data")
        
        s1,graph1,s2=model_prediction(df)

        return render_template('graph.html',string1=s1,string2=s2,graph1=graph1)
    
    return render_template("index.html")  

if __name__ =="__main__":  
    app.run(debug = True)  