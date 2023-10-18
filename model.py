#라이브러리 사용 선언
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.utils import *
from sklearn.preprocessing import *
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
import os

#상품 데이터 불러오기
good = pd.read_csv('./good_result2.csv', encoding='CP949')

#날짜 데이터 불러오기
date = pd.read_csv('./date.csv')

# 날짜 인덱스 저장
Index=date['date']

#상품 데이터에 날짜 데이터를 인덱스로 지정
good.set_index(Index,inplace = True)

# 환율, 국제유가, 확진자 데이터 불러오기
df = pd.read_csv('./deeplearningfeatures.csv')
df.set_index(Index, inplace=True)

learning_data=pd.concat([df,good[input_data]],axis=1)

#랜덤 시드 적용
tf.random.set_seed(3)

#데이터 정규화(0~1)
learning_data.sort_index(ascending=False).reset_index(drop=True)
scaler = MinMaxScaler()
scale_cols = learning_data.columns
ld_scaled = scaler.fit_transform(learning_data[scale_cols])
ld_scaled = pd.DataFrame(ld_scaled)
ld_scaled.columns = scale_cols

def md(d, l, ws):
    fl = []
    ll = []
    for i in range(len(d) - ws):
        fl.append(np.array(d.iloc[i:i+ws]))
        ll.append(np.array(l.iloc[i+ws]))
    return np.array(fl), np.array(ll)

fc = learning_data.columns[:-1]
lc = learning_data.columns[-1:]

tf = ld_scaled[fc]
tl = ld_scaled[lc]

tf, tl = md(tf, tl, 12)

x_train, x_valid, y_train, y_valid = train_test_split(tf, tl, test_size=0.2)

#모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=(12, 10), return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='relu'))

#손실함수, 최적화 함수, 요약
model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', patience=10)

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model.fit(x_train, y_train, 
                                    epochs=100, 
                                    batch_size=5,
                                    validation_data=(x_valid, y_valid), 
                                    callbacks=[early_stop, checkpoint])

model.load_weights(filename)
pred = model.predict(tf)

plt.figure(figsize=(12, 9))
plt.plot(tl, label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.savefig(str(good.columns[input_data])+'.png')