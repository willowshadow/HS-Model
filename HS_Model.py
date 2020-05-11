from keras.preprocessing import sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.layers import LeakyReLU,Softmax,ReLU
from keras.layers import Dense, Embedding, LSTM,SpatialDropout1D, Flatten,Dropout,Activation
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from nltk.corpus import stopwords
import numpy as np
from numpy import where
import matplotlib.pyplot as plt
from sklearn.utils import class_weight


Data=pd.read_csv("hate_speech.csv")
# split into input (X) and output (Y) variables

print(Data['label'].value_counts(10))

print(Data.head())

df = Data.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
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
    text = text.replace('rt', ' ')
    text = re.sub(r'\W+', ' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def clean_tweet(tweet): 
    ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
    tweet =' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 

    return tweet

df['post'].apply(clean_tweet)

print(df.head())


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 5000

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 100
# This is fixed.
EMBEDDING_DIM = 64
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
#tokenizer.fit_on_texts(df['post'])


tokenizer.fit_on_texts(df['post'])
X = tokenizer.texts_to_sequences(df['post'])
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

print('Shape of data tensor:', X.shape)



#Y=np.array(Y).reshape(-2,1)
Y=df['label']
#Y=where(Y == 0,-1,1)
#print(Y)
#Y = to_categorical(df['label'])
print('Shape of label tensor:', Y.shape)


print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 32)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#model = Sequential()
#model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
#model.add(SpatialDropout1D(0.2))
#model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))

#model.add(LeakyReLU(alpha=0.01))

#model.add(LSTM(64,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))

#model.add(LeakyReLU(alpha=0.01))

#model.add(LSTM(32,dropout=0.2,recurrent_dropout=0.2))
#model.add(Dense(2,activation='softmax'))

class_weights = {0: 1.,
                1: 8.}

#y_ints = [y.argmax() for y in Y_train]
#print(y_ints)
#class_weights = class_weight.compute_class_weight('balanced',
#                                                 np.unique(y_ints),
#                                                 y_ints)
#class_weight_dict = dict(enumerate(class_weights))


embedding_vecor_length = 128
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, embedding_vecor_length, input_length=MAX_SEQUENCE_LENGTH,trainable=True))
#model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=4))
#model.add(Flatten())
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True,activation='relu'))
model.add(Dense(256))
model.add(Dropout(0.50))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(ReLU())
model.add(Dense(256))
model.add(Dropout(0.50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
print(model.summary())


#model.add(LeakyReLU(alpha=0.05))



epochs = 25
batch_size = 128

early_stop = EarlyStopping(monitor="val_loss", patience=5)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2,
	patience=2, min_lr=0.0001)

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,class_weight=class_weights,validation_data=(X_test, Y_test),callbacks=[early_stop,reduce_lr],shuffle=False)
#callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)]


accr = model.evaluate(X_test,Y_test,use_multiprocessing=True)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))



plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
model.save("Model.h5",overwrite=True,include_optimizer=True)


# below manual test gives 0 always despite changing text to negative sentiment
#new_complaint = ['This man is an asshole, screw him.']
#new_complaint=tokenizer.fit_on_texts(new_complaint)
#seq = tokenizer.texts_to_sequences(new_complaint)
#padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
#pred = model.predict_proba(padded)

#labels = ['Not hate',' Hate']
##print(pred[0])
#print(pred)

i=1
while (int(i)!=0):
    new = input("Enter text to check")
    new=clean_text(new)
    tokenizer.fit_on_texts(new)
    seq = tokenizer.texts_to_sequences(new)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict_proba([padded])
    pred2 = model.predict_classes([padded])
    labels = ['Not',' Offensive']
    print(pred)
    print(pred2)
    i=input("0 or else")
    int(i)