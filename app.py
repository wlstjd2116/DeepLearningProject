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
import deeplearning_model
import json
import plotly
import plotly.graph_objs as go

app = Flask(__name__)

tf.random.set_seed(3)

model = load_model('prediction_model.h5')
date = pd.read_csv('./2021projectdata/date.csv')
good = pd.read_csv('./2021projectdata/good_result2.csv', encoding='CP949')
df = pd.read_csv('./2021projectdata/deeplearningfeatures.csv')
bar = None
@app.route('/')
def main():
    return render_template("main.html")

@app.route('/project')
def project():
    return render_template("project.html")


@app.route('/search', methods=["GET"])
def search(select_good=None):
    if request.method == 'GET':
        temp = request.args.get('select_good')
        if temp:
            good_price = pd.concat([date['date'], good[temp]],axis=1)
            return render_template("search.html", select_good=temp,  tables=[good_price.to_html(index=False, justify='center')], titles=good_price.columns.values)
        return render_template("search.html", select_good=temp)

@app.route('/predict', methods=["GET"])
def predict(predict_good=None):
    global temp3, bar
    if request.method == 'GET':
#         er_open, er_end, er_high, er_low, consumer, oil_dubai, oil_brent, oil_WTI, oil_oman, patients = False, False, False, False, False, False, False, False, False, False
#         if request.args.get('er_open'):
#             er_open  = True
#             df_er_open = df['er_open']
#             df2 = pd.concat([df2, df_er_open], axis=1)

#         if request.args.get('er_end'):
#             er_end = True
#             df_er_end = df['er_end']
#             df2 = pd.concat([df2, df_er_end], axis=1)

#         if request.args.get('er_high'):
#             er_high = True
#             df_er_high = df['er_high']
#             df2 = pd.concat([df2, df_er_high], axis=1)

#         if request.args.get('er_low'):
#             er_low = True
#             df_er_low = df['er_low']
#             df2 = pd.concat([df2, df_er_low], axis=1)

#         if request.args.get('consumer'):
#             consumer = True
#             df_consumer = df['consumer']
#             df2 = pd.concat([df2, df_consumer], axis=1)

#         if request.args.get('oil_dubai'):
#             oil_dubai = True
#             df_oil_dubai = df['oil_dubai']
#             df2 = pd.concat([df2, df_oil_dubai], axis=1)

#         if request.args.get('oil_brent'):
#             oil_brent = True
#             df_oil_brent = df['oil_brent']
#             df2 = pd.concat([df2, df_oil_brent], axis=1)

#         if request.args.get('oil_WTI'):
#             oil_WTI = True
#             df_oil_WTI = df['oil_WTI']
#             df2 = pd.concat([df2, df_oil_WTI], axis=1)

#         if request.args.get('oil_oman'):
#             oil_oman = True
#             df_oil_oman = df['oil_oman']
#             df2 = pd.concat([df2, df_oil_oman], axis=1)

#         if request.args.get('patients'):
#             patients = True
#             df_patients = df['patients']
#             df2 = pd.concat([df2, df_patients], axis=1)

        temp3 = request.args.get('predict_good')
        if temp3:
        #    if os.path.isfile(temp3+".h5"):
       #         model=load_model(str(temp3)+".h5")
       #         bar = model.predict(
        #    else:
            bar = deeplearning_model.learning(temp2=temp3, df=df, good=good, date=date) # return graphJSON
        return render_template("predict.html", predict_good=temp3, plot=bar)#, er_open=er_open, er_end=er_end, er_high=er_high, er_low=er_low, consumer=consumer, oil_dubai=oil_dubai, oil_brent=oil_brent, oil_WTI=oil_WTI, oil_oman=oil_oman, patients=patients)
    else:
        return render_template("predict.html")
if __name__=='__main__':

    app.run(debug=True)