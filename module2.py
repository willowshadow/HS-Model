import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,MaxPooling1D,Conv1D,Flatten,GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from keras import optimizers as OPT
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import re
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import pickle
import nltk
from nltk.corpus import stopwords
import nltk.stem as Stem
pd.set_option('display.max_colwidth', -1)
plt.style.use('ggplot')






REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = nltk.corpus.stopwords.words('english')
#STOPWORDS.extend(['nigga','bitch','niguah','niggah'])

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = clean_tweet(text)
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('rt', ' ')
    text = re.sub(r'\d+', '', text)
    text = text.replace('&#', ' ')
    text = re.sub(r'\W+', ' ', text)
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    text = text.split()
    stemm = Stem.SnowballStemmer("english")
    stemm = [stemm.stem(word) for word in text]
    text = " ".join(stemm)
    return text

def clean_tweet(tweet): 
    ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
    tweet =' '.join(re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 

    return tweet



batch_size = 512
# also adding weights


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    #plt.ylim(0,1)
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

path = "hate_speech.csv" # change to the path to the Sentiment.csv file
data = pd.read_csv(path)
print(data.head())
data= data.reset_index(drop=True)



path2="Cleaned_Dataset3.csv"
data1=pd.read_csv(path2,encoding = 'utf-8',sep=',')

path1="labeled_data.csv"
data_test=pd.read_csv(path1)
data_test=data_test.reset_index(drop=True)


#data1.drop(data1[data1.label=='abusive'].index,inplace = True)

#data1.drop(data1[data1.label=='spam'].index,inplace = True)

#data1.label.replace(['hateful','normal'],[1,0],inplace=True)



data1.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
print(data1.describe())
#data1=data1.reset_index(drop=True)
#print(data1.head(100))

#data1['post']=data1['post'].apply(clean_text)
#print(data1.head(100))


#data1.to_csv(r'Cleaned_Dataset3.csv',index=False)                      


#exit()
#Validation dataset



A=1
B=0
data1.label.replace([1,0],[A,B],inplace=True)

print(data1.head(6))

# Separate majority and minority classes


#data_test['label']=data_test[data_test.label != 1]
#data_test.drop(data_test[data_test.label == 1].index, inplace=True)

data_test.label.replace([2,1,0], [0,0,1], inplace=True)
data_test['label'].hist()
#data_test['label']=data_test['label'].map({2:0,0:1})

print(data_test['label'].head(20))
print(data[data.label==0])

data['post']=data['post'].apply(clean_text)
data_test['post']=data_test['post'].apply(clean_text)

print(data_test['post'].head(100))

print(data_test['label'].value_counts())




#exit()

data_majority = data[data['label'] == 0]
data_minority = data[data['label'] == 1]

data_majority1 = data1[data1['label'] == 0]
data_minority1 = data1[data1['label'] == 1]

data_majority2 = data_test[data_test['label'] == 0]
data_minority2 = data_test[data_test['label'] == 1]

print(data_minority)
print(data_majority)

print(data_minority1)
print(data_majority1)

print(data_minority2)
print(data_majority2)


# will be used later in defining class weights
bias = data_minority.shape[0]/data_majority.shape[0]
bias1 = data_minority1.shape[0]/data_majority1.shape[0]
bias2 = data_minority2.shape[0]/data_majority2.shape[0]


class_weights = {0: 1 ,
                 1: 1/bias }
class_weights1 = {0: 1 ,
                 1: 1/bias1 }
class_weights2 = {0: 1 ,
                 1: 1.6/bias2 }


# lets split train/test data first then 
train = pd.concat([data_majority.sample(frac=0.2,random_state=200),
         data_minority.sample(frac=0.2,random_state=200)])
test = pd.concat([data_majority.drop(data_majority.sample(frac=0.2,random_state=200).index),
        data_minority.drop(data_minority.sample(frac=0.2,random_state=200).index)])

train1 = pd.concat([data_majority1.sample(frac=0.2,random_state=200),
         data_minority1.sample(frac=0.2,random_state=200)])
test1 = pd.concat([data_majority1.drop(data_majority1.sample(frac=0.2,random_state=200).index),
        data_minority1.drop(data_minority1.sample(frac=0.2,random_state=200).index)])

train2 = pd.concat([data_majority2.sample(frac=0.75,random_state=200),
         data_minority2.sample(frac=0.75,random_state=200)])
test2 = pd.concat([data_majority2.drop(data_majority2.sample(frac=0.75,random_state=200).index),
        data_minority2.drop(data_minority2.sample(frac=0.75,random_state=200).index)])

train = shuffle(train)
test = shuffle(test)

train1 = shuffle(train1)
test1 = shuffle(test1)

train2 = shuffle(train2)
test2 = shuffle(test2)

print('negative data in training:',(train.label == 1).sum())
print('positive data in training:',(train.label == 0).sum())
print('negative data in test:',(test.label == 1).sum())
print('positive data in test:',(test.label == 0).sum())

data_majority = train[train['label'] == 0]
data_minority = train[train['label'] == 1]

print("majority class before upsample:",data_majority.shape)
print("minority class before upsample:",data_minority.shape)




print('negative data in training:',(train1.label == 1).sum())
print('positive data in training:',(train1.label == 0).sum())
print('negative data in test:',(test1.label == 1).sum())
print('positive data in test:',(test1.label == 0).sum())

data_majority1 = train1[train1['label'] == 0]
data_minority1 = train1[train1['label'] == 1]

print("majority class before upsample:",data_majority1.shape)
print("minority class before upsample:",data_minority1.shape)


print('negative data in training:',(train2.label == 1).sum())
print('positive data in training:',(train2.label == 0).sum())
print('negative data in test:',(test2.label == 1).sum())
print('positive data in test:',(test2.label == 0).sum())

data_minority2 = train2[train2['label'] == 1]
data_majority2 = train2[train2['label'] == 0]

print("majority class before upsample:",data_majority2.shape)
print("minority class before upsample:",data_minority2.shape)
# Upsample minority class
data_minority_upsampled = resample(data_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples= data_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results

data_minority_upsampled1 = resample(data_minority1, 
                                 replace=True,     # sample with replacement
                                 n_samples= data_majority1.shape[0],    # to match majority class
                                 random_state=123) # reproducible results

data_minority_upsampled2 = resample(data_minority2, 
                                 replace=True,     # sample with replacement
                                 n_samples= data_majority2.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled])

data_upsampled1 = pd.concat([data_majority1, data_minority_upsampled1])

data_upsampled2 = pd.concat([data_majority2, data_minority_upsampled2])


#data_upsampled1.to_csv(r'UP-Sampled-1.csv', index = False)
 
# Display new class counts
print("After upsampling\n",data_upsampled.label.value_counts(),sep = " ")
print("After upsampling\n",data_upsampled1.label.value_counts(),sep = " ")
print("After upsampling\n",data_upsampled2.label.value_counts(),sep = " ")
maxlen=64

max_fatures = 3000
tokenizer = Tokenizer(num_words=max_fatures,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True,split=' ')

#tokenizer.fit_on_texts(data['post'].values) # training with whole data

#X_train = tokenizer.texts_to_sequences(train['post'].values)
#X_train = pad_sequences(X_train,maxlen=maxlen,padding='post')
#Y_train = pd.DataFrame(train['label']).values
#print('x_train shape:',X_train.shape)

#X_test = tokenizer.texts_to_sequences(test['post'].values)
#X_test = pad_sequences(X_test,maxlen=maxlen,padding='post')
#Y_test = pd.DataFrame(test['label']).values
#print("x_test shape", X_test.shape)


#Val_X=tokenizer.texts_to_sequences(data_test['post'].values)
#Val_X=pad_sequences(Val_X,maxlen=maxlen,padding='post')
#Val_Y=data_test['label'].values

#tokenizer.fit_on_texts(data1['post'].values) # training with whole data

#X_train = tokenizer.texts_to_sequences(train1['post'].values)
#X_train = pad_sequences(X_train,maxlen=maxlen,padding='post')
#Y_train = pd.DataFrame(train1['label']).values
#print('x_train shape:',X_train.shape)

#X_test = tokenizer.texts_to_sequences(test1['post'].values)
#X_test = pad_sequences(X_test,maxlen=maxlen,padding='post')
#Y_test = pd.DataFrame(test1['label']).values
#print("x_test shape", X_test.shape)


#Val_X=tokenizer.texts_to_sequences(data_test['post'].values)
#Val_X=pad_sequences(Val_X,maxlen=maxlen,padding='post')
#Val_Y=data_test['label'].values

tokenizer.fit_on_texts(data_test['post'].values) # training with whole data

X_train = tokenizer.texts_to_sequences(train2['post'].values)
X_train = pad_sequences(X_train,maxlen=maxlen,padding='post')
Y_train = pd.DataFrame(train2['label']).values
print('x_train shape:',X_train.shape)

X_test = tokenizer.texts_to_sequences(test2['post'].values)
X_test = pad_sequences(X_test,maxlen=maxlen,padding='post')
Y_test = pd.DataFrame(test2['label']).values
print("x_test shape", X_test.shape)


Val_X=tokenizer.texts_to_sequences(data1['post'].values)
Val_X=pad_sequences(Val_X,maxlen=maxlen,padding='post')
Val_Y=data1['label'].values

Val_X1=tokenizer.texts_to_sequences(data['post'].values)
Val_X1=pad_sequences(Val_X1,maxlen=maxlen,padding='post')
Val_Y1=data['label'].values


#+=================================


trainX = tokenizer.texts_to_sequences(train['post'].values)
trainX = pad_sequences(trainX,maxlen=maxlen,padding='post')
trainY = pd.DataFrame(train['label']).values
print('x_train shape:',trainX.shape)

testX = tokenizer.texts_to_sequences(test['post'].values)
testX = pad_sequences(testX,maxlen=maxlen,padding='post')
testY = pd.DataFrame(test['label']).values
print("x_test shape", testX.shape)

#=================================
train1X = tokenizer.texts_to_sequences(train1['post'].values)
train1X = pad_sequences(train1X,maxlen=maxlen,padding='post')
train1Y = pd.DataFrame(train1['label']).values
print('x_train shape:',trainX.shape)

test1X = tokenizer.texts_to_sequences(test1['post'].values)
test1X = pad_sequences(test1X,maxlen=maxlen,padding='post')
test1Y = pd.DataFrame(test1['label']).values
print("x_test shape", test1X.shape)



#X_train = X_train.reshape(-1, 1, 32)
#X_test  = X_test.reshape(-1, 1, 32)
#Y_train = Y_train.reshape(-1, 1, 1)
#Y_test = Y_test.reshape(-1, 1, 1)


##print(X_test)
##print(Y_test)
##print(X_train)
##print(Y_train)

#Y_train=pd.DataFrame(Y_train)
#print(Y_train.head(100))


#exit()

#print(Y_test)
#X = tokenizer.texts_to_sequences(data['post'].values)
#X = pad_sequences(X, maxlen=maxlen,padding='post')
#word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))

#print('Shape of data tensor:', X.shape)



##Y=np.array(Y).reshape(-2,1)
#Y=(data['label'].values)
##Y=where(Y == 0,-1,1)
##print(Y)
##Y = to_categorical(df['label'])
#print('Shape of label tensor:', Y.shape)


#print(X)
#print(Y)

#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 222)


vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(vocab_size)
embed_dim = 32
lstm_out = 128

model = Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=embed_dim,input_length = maxlen))
model.add(SpatialDropout1D(0.5))
#model.add(Flatten())
model.add(Conv1D(filters=128, kernel_size=4, padding='same',activation='relu'))

model.add(MaxPooling1D(pool_size=4))

#model.add(Dense(128))
#model.add(SpatialDropout1D(0.5))

model.add(LSTM(128, dropout=0.4,return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer=OPT.Adam(),metrics = ['accuracy'])
print(model.summary())

early_stop = EarlyStopping(monitor="val_loss", patience=10)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2,
	patience=2, min_lr=0.000001)



history=model.fit(X_train, Y_train, epochs = 50, batch_size=batch_size, verbose = 1,class_weight=class_weights2,validation_data=(X_test, Y_test),callbacks=[early_stop,reduce_lr],shuffle=False)
#history1=model.fit(train1X, train1Y, epochs = 15, batch_size=batch_size, verbose = 1,class_weight=class_weights1,validation_data=(test1X, test1Y),callbacks=[early_stop,reduce_lr],shuffle=False)
#history2=model.fit(trainX, trainY, epochs = 15, batch_size=batch_size, verbose = 1,class_weight=class_weights,validation_data=(testX, testY),callbacks=[early_stop,reduce_lr],shuffle=False)

accr = model.evaluate(X_test,Y_test,use_multiprocessing=True)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

loss, accuracy = model.evaluate(X_train, Y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)

#loss, accuracy = model.evaluate(X_train, Y_train, verbose=1)
#print("Training Accuracy: {:.4f}".format(accuracy))
#loss, accuracy = model.evaluate(test1X, test1Y, verbose=1)
#print("Testing Accuracy:  {:.4f}".format(accuracy))
#plot_history(history1)

#loss, accuracy = model.evaluate(X_train, Y_train, verbose=1)
#print("Training Accuracy: {:.4f}".format(accuracy))
#loss, accuracy = model.evaluate(testX, testY, verbose=1)
#print("Testing Accuracy:  {:.4f}".format(accuracy))
#plot_history(history2)





#plt.title('Accuracy')
#plt.plot(history.history['accuracy'], label='train')
#plt.plot(history.history['val_accuracy'], label='test')
#plt.legend()
#plt.show()

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model train vs validation loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper right')
#plt.show





Y_pred = model.predict_classes(X_test,batch_size = batch_size)
Y_pred2=model.predict_classes(test1X,batch_size=batch_size)
Y_pred3=model.predict_classes(testX,batch_size=batch_size)


print(Y_pred)
print(Y_test)
print(Y_pred.tolist())

print(X_test)

print(Y_test.tolist())

print(confusion_matrix(Y_test,np.asarray(Y_pred)))
print(confusion_matrix(test1Y,np.asarray(Y_pred2)))
print(confusion_matrix(testY,np.asarray(Y_pred3)))

print(Y_pred.flatten())
df_test = pd.DataFrame({'true': Y_test.flatten(), 'pred':Y_pred.flatten()})
#df_test.to_csv(r'preds.csv',index=False)
#print(df_test.head())

#df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))

print(df_test.head())

#print("confusion matrix",confusion_matrix(df_test.true, df_test.pred))

print(classification_report(Y_test.flatten(), Y_pred.flatten()))
print(classification_report(test1Y.flatten(), Y_pred2.flatten()))
print(classification_report(testY.flatten(), Y_pred3.flatten()))

save=input("Want to save model and tokenizer ?, enter 1 for yes 0 for no")

if(int(save)==1):
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model.save("Model-2.h5",overwrite=True,include_optimizer=True)

#======================================================================



