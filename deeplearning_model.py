from flask import Flask, render_template, request, url_for
import flask
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import *
from keras.models import *
from keras.utils import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from keras.datasets import mnist
from numpy import argmax
import plotly
import plotly.graph_objs as go
import json
def create_plot(real, y):

    x = np.linspace(0, len(y),len(y))
    y = y
    real = real
    df = pd.DataFrame({'x': x, 'y': y, 'real' : real}) # creating a sample dataframe


    data = [
        go.Line(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y'], mode = 'lines+markers', name = 'predict_price'
        ),go.Line(
            x=df['x'],
            y=df['real'], mode = 'lines+markers', name = 'real_price')
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def md(d, l, ws):
        fl = []
        ll = []
        for i in range(len(d) - ws):
            fl.append(np.array(d.iloc[i:i+ws]))
            ll.append(np.array(l.iloc[i+ws]))
        return np.array(fl), np.array(ll)

def learning(temp2, df, good, date):
        idx=temp2
        learning_data = pd.concat([df, good[idx]],axis=1)
        learning_data.set_index(date['date'],inplace=True)
        learning_data.sort_index(ascending=False).reset_index(drop=True)

        scaler = MinMaxScaler()
        scale_cols = learning_data.columns
        ld_scaled = scaler.fit_transform(learning_data[scale_cols])
        orig_ld_scaled = scaler.inverse_transform(ld_scaled)
        ld_scaled = pd.DataFrame(ld_scaled)
        ld_scaled.columns = scale_cols
        if len(learning_data.columns) > 1:
            fc = learning_data.columns[:-1] # 특성
            lc = learning_data.columns[-1:] # 생필품
            tf = ld_scaled[fc]
            tl = ld_scaled[lc]     
            tf, tl = md(tf, tl, 12)
            x_train, x_valid, y_train, y_valid = train_test_split(tf, tl, test_size=0.2)
            model = Sequential()
            model.add(LSTM(20, input_shape=(12, learning_data.shape[1]-1), return_sequences=True))
            model.add(LSTM(20, return_sequences=False))
            model.add(Dense(1, activation='relu'))
            model.compile(loss='mean_squared_error', optimizer='adam')
            early_stop = EarlyStopping(monitor='val_loss', patience=10)
            model_path = 'model'
            filename = os.path.join(model_path, 'tmp_checkpoint.h5')
            checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            history = model.fit(x_train, y_train, epochs=500, batch_size=10, validation_data=(x_valid, y_valid), callbacks=[early_stop, checkpoint])
        else:
            fc = learning_data
            lc = learning_data
            tf = ld_scaled[fc]
            tl = ld_scaled[lc]        
            tf, tl = md(tf, tl, 12)
            x_train, x_valid, y_train, y_valid = train_test_split(tf, tl, test_size=0.2)
            model = Sequential()
            model.add(LSTM(20, input_shape=(12, 1), return_sequences=True))
            model.add(LSTM(20, return_sequences=False))
            #model.add(dropout(0.2))
            model.add(Dense(1, activation='relu'))
            model.compile(loss='mean_squared_error', optimizer='adam')
            early_stop = EarlyStopping(monitor='val_loss', patience=10)
            model_path = 'model'
            filename = os.path.join(model_path, 'tmp_checkpoint.h5')
            checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            history = model.fit(x_train, y_train, epochs=500, batch_size=10, validation_data=(x_valid, y_valid), callbacks=[early_stop, checkpoint])            

        model.load_weights(filename)
        pred = model.predict(tf)
        #pred = scaler.inverse_transform(pred)
        #plt.figure(figsize=(12, 10))
        #plt.plot(tl, label = 'actual')
        #plt.plot(pred, label = 'prediction')
        #plt.legend()
        #plt.savefig('static/image/'+idx+'.png')
        graphJSON = create_plot(tl.flatten(), pred.flatten())
        model.save(idx+".h5")
        return graphJSON