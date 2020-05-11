
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM , MaxPooling1D
from keras.optimizers import Adam
from keras.datasets import imdb
from keras.optimizers import *
from keras.activations import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tk = Tokenizer()

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features,128))
#model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=4))
#model.add(Flatten())
model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1, activation='hard_sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# try using different optimizers and different optimizer configs


print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


i=1
while (int(i)!=0):
    new = input("Enter text to check")
    #new=clean_text(new)
    tk.fit_on_texts(new)
    index_list = tk.texts_to_sequences(new)
    new = pad_sequences(index_list, maxlen=maxlen)
    pred = model.predict_proba(new)
    pred2 = model.predict_classes(new)
    labels = ['Not',' Offensive']
    print(np.max(pred))
    print(pred2)
    i=input("0 or else")
    int(i)