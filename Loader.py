
from keras.models import load_model
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from nltk.corpus import stopwords
#import HS_Model
import re
import sqlite3
from keras.backend import manual_variable_initialization 
import pickle
from keras.utils.vis_utils import plot_model
import nltk
from nltk.corpus import stopwords
import nltk.stem as Stem

manual_variable_initialization(True)




REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.extend(['nigga','bitch'])

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    #text = clean_tweet(text)
    print(text,'\n')
    text = text.lower() # lowercase text
    print(text,'\n')
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    print(text,'\n')
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    print(text,'\n')
    text = text.replace('rt', ' ')
    print(text,'\n')
    text = re.sub(r'\d+', '', text)
    text = text.replace('&#', ' ')
    print(text,'\n')
    text = re.sub(r'\W+', ' ', text)
    #text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    text = text.split()
    lemm = Stem.SnowballStemmer("english")
    lemm_words = [lemm.stem(word) for word in text]
    text = " ".join(lemm_words)
    print(text,'\n')
    return text

def clean_tweet(tweet): 
    ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
    tweet =' '.join(re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 

    return tweet

new_complaint = "@SabnamFatima All Terrorist are Muslim they kill innocent people like kasab they killed all front on their eyes, but veer nathuram godse killed only mahatma gandi, #MuslimTerrorist"


#newnew_complaint= clean_text(new_complaint)
#tokenizer = Tokenizer(num_words=2000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
model = load_model('Model-2.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
#model.compile(optimizer = 'adamax', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model1=keras.models.clone_model(model)

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True,expand_nested=True,dpi=200)



PP=clean_text(new_complaint)
print(PP)

exit()
#i=1
#int(i)
#while (int(i)!=0):
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
    #new = input("Enter text to check")
    #new=clean_text(new)

    #tokenizer.fit_on_texts(new)
    #tokenizer.fit_on_texts(new)
    #seq = tokenizer.texts_to_sequences(new)
    #padded = pad_sequences(seq, maxlen=100,)
    #padded=np.array(padded)
    #pred = model.predict_proba(padded,batch_size=32)
    #pred2 = model.predict_classes([padded],batch_size=32)
    #pred3 = model.predict([padded],batch_size=32,verbose=1)
    #labels = ['Hate',' Offensive','Neutral']
    #print(pred)
    #print(pred2)
    #print(pred3)

    #class_pred_prob= pred/pred2
 
    #class1=0
    #class2=1
    #class3=2
    #a=len(np.max(pred,axis=1))
    #while (int(a)>0):
    #    print("abc")
    #    a-=1


i=1
while (int(i)!=0):
    twt = input("Enter text to check")
    twt=clean_text(twt)
    print(twt)
    #twt = ['keep up the good work nigga ']
    #vectorizing the tweet by the pre-fitted tokenizer instance
    twt = tokenizer.texts_to_sequences([twt])
    print(twt)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=64, padding='post')
    print(twt)
    sentiment = model.predict_proba(twt,batch_size=1,verbose = 1)
    classes = model.predict_classes(twt,batch_size=1,verbose = 1)
    print(sentiment)
    print(classes)
    if(np.argmax(sentiment) == 0):
        print("positive")
    elif (np.argmax(sentiment) == 1):
        print("negative")
    i=input("0 or else")
    int(i)

#model.summary()
#import onnxmltools

## Update the input name and path for your Keras model
#input_keras_model = 'Model.h5'

## Change this path to the output name and path for the ONNX model
#output_onnx_model = 'model.onnx'

## Load your Keras model
#keras_model = load_model(input_keras_model)

## Convert the Keras model into ONNX
#onnx_model = onnxmltools.convert_keras(keras_model)

## Save as protobuf
#onnxmltools.utils.save_model(onnx_model, output_onnx_model)
#print(new_complaint)
##text=clean_text(new_complaint)
##print(text)
#tokenizer = Tokenizer(num_words=20000)
#tokenizer.fit_on_texts(new_complaint)
##print(text)
#seq = tokenizer.texts_to_sequences(new_complaint)
#padded = pad_sequences(seq, maxlen=200)
#print(padded)
#pred = model.predict_proba(padded)
#print((pred))
#labels = ['Hate',' Offensive','Neutral']
#print(pred, labels[np.argmax(pred)])

#seq = tokenizer.texts_to_sequences("What the fuck is wrong with trump he ssucks")
#padded = pad_sequences(seq, maxlen=200)
#pred = model.predict_proba(padded,batch_size=8,verbose=2)
#labels = ['Hate',' Offensive','Neutral']
#print(pred)

#seq = tokenizer.texts_to_sequences(["All black people are bad"])
#padded = pad_sequences(seq, maxlen=200)
#pred = model.predict_proba(padded)
#labels = ['Hate',' Offensive','Neutral']
#print(pred, labels[np.argmax(pred)])









#=====================================================
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
 
    return conn


def update_task(conn, task):
    """
    update priority, begin_date, and end date of a task
    :param conn:
    :param task:
    :return: project id
    """
    sql = ''' UPDATE Tweets
              SET pred = ? 
              WHERE ID = ?'''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()

def update_task2(conn, task):
    """
    update priority, begin_date, and end date of a task
    :param conn:
    :param task:
    :return: project id
    """
    sql = ''' UPDATE Tweets
              SET Class = ? 
              WHERE ID = ?'''
    cur = conn.cursor()
    cur.execute(sql, task)
    conn.commit()


def predictor(text):
    twt=clean_text(text)
    print(twt)
    #twt = ['keep up the good work nigga ']
    #vectorizing the tweet by the pre-fitted tokenizer instance
    twt = tokenizer.texts_to_sequences([twt])
    print(twt)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=30, padding='post')
    print(twt)
    sentiment = model.predict_proba(twt,batch_size=1,verbose = 1)
    classes = model.predict_classes(twt,batch_size=1,verbose = 1)
    print(sentiment[0][0])
    print(classes[0][0])

    return (((sentiment[0][0])*100)),int(classes[0][0])

database = r"C:\Users\Danyal Tariq\source\repos\HS-Model\Donald trump.db"
 

    # create a database connection
conn = create_connection(database)


cursor = conn.cursor()
print("Connected to SQLite")

cursor = cursor.execute('select * from Tweets;')
i=len(cursor.fetchall())
print(i,"rows")

while i>0:
    sqlite_select_query = """SELECT * from Tweets where ID = ?"""
    cursor.execute(sqlite_select_query, (i,))
    print("Reading single row \n")
    record = cursor.fetchone()
    print(record)
    #print("Id: ", record[0])
    #print("text: ", record[1])
    #print("pred: ", record[2])
    Prediction,Classvar=predictor(record[1])
    update_task(conn,(Prediction,i))
    update_task2(conn,(Classvar,i))
    #print("prediction: ", record[3])
    i-=1

cursor.close()

#with conn
#update_task(conn, (95,1))
