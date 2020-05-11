import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords

from bs4 import BeautifulSoup

import sys
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adagrad, Adam, Adamax
from keras.layers.convolutional import MaxPooling1D
from keras.activations import *

import matplotlib.pyplot as plt
import textblob

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 8000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1


data_train = pd.read_csv('labeled_data.csv',sep=',')
print(data_train['label'].value_counts(10))

data_train = data_train.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@_,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
RT=re.compile('rt')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = clean_tweet(text)
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('rt', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def clean_tweet(tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        tweet =' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 

        return tweet



print (data_train.shape)
print (data_train['post'].head())

data_train['post'] = data_train['post'].apply(clean_text)
#clean_str(data_train['post'].values)
print (data_train.head())
#texts = []
#labels = []



data_train['label'] = ((data_train['label']))
print('Shape of label tensor:', len(data_train['label']))
Y=data_train['label']

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data_train['post'].values)
print(data_train['post'].head(1))
sequences = tokenizer.texts_to_sequences(data_train['post'].values)
print(sequences)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(data)
X=np.array(data)

#sys.exit()
Y=to_categorical(data_train['label'],num_classes=None)
Y=np.array(Y)

print(X.shape)
print(Y.shape)

dum_df = pd.get_dummies(data_train, columns=["label"], prefix=["_"] )
print(dum_df.head())
print(data_train.head())
#Y=np.array(dum_df).reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 32)

#==============================Archi-1==============================
#sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
##embedded_sequences = embedding_layer(sequence_input)
#l_gru = LSTM(100, return_sequences=False)(sequence_input)
#dropout_1=Dropout(0.5)(l_gru)
#dense_1 = Dense(100,activation='tanh')(dropout_1)
#dropout_2=Dropout(0.5)(dense_1)
#dense_2 = Dense(3, activation='softmax')(dropout_2)

#model = Model(sequence_input, dense_2)
#=============================================================

embedding_vecor_length = 128
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, embedding_vecor_length, input_length=MAX_SEQUENCE_LENGTH))
#model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='hard_sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001,amsgrad=True), metrics=['accuracy'])


#model.compile(loss='categorical_crossentropy',
#              optimizer='rmsprop',
#              metrics=['acc'])

model.summary()

history = model.fit(X_train, Y_train, epochs=50, batch_size=16,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=8, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test,use_multiprocessing=True,workers=5)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

#model.save("Model-1.h5",overwrite=True,include_optimizer=True)

i=1
int(i)
while (int(i)!=0):
    #(new_complaint) = "This man is an asshole, screw him."
    #np.array(new_complaint)
    #print(new_complaint)
    #text=new_complaint
    #print(text)
    #tokenizer = Tokenizer(num_words=20000)
    #tokenizer.fit_on_texts(text)
    #print(text)
    #seq = tokenizer.texts_to_sequences(text)
    #print(seq)
    #padded = pad_sequences(seq, maxlen=100)
    #print(padded)
    #pred = model.predict(padded)
    #print((np.sum(pred,axis=0)))
    #labels = ['Hate',' Offensive','Neutral']
    #print(pred)
    #seq = tokenizer.texts_to_sequences(new_complaint)
    #padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    #pred = model.predict_classes(padded)
    #labels = ['Hate',' Offensive','Neutral']
    ##print(pred[0])
    #print(pred, labels[np.argmax(pred)])
    new = input("Enter text to check")
    new=clean_text(new)

    tokenizer.fit_on_texts(new)
    seq = tokenizer.texts_to_sequences(new)
    np.ndarray(seq)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    padded=np.array(padded)
    pred = model.predict_proba(padded,batch_size=8)
    pred2 = model.predict_classes(padded,batch_size=8)
    labels = ['Hate',' Offensive','Neutral']
    print(pred,(np.argmax(np.sum(pred,axis=0))))
    print(pred2)

    #class_pred_prob= pred/pred2

   
    class1=0
    class2=1
    class3=2
    a=len(np.max(pred,axis=1))
    while (int(a)>0):
        print("abc")
        a-=1



i=input("Enter 0 to continue")
int(i)
