# Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , LSTM , Embedding , Conv1D , Bidirectional , GRU , Dropout
from keras.layers import GlobalMaxPool1D, Dropout, Activation,CuDNNLSTM
from keras.layers import MaxPooling1D, BatchNormalization,Conv2D,Flatten


# Training and Test set processing
'''
text,target
book a restaurant,book_restaurant

'''
train = pd.read_csv('../input/train.csv')  
test = pd.read_csv('../input/test.csv')

embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

train_X = train["text"].fillna("_na_").values

test_X = test["text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)

test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train['target'].values
print("Done")



#Using Embeddings
embedding_index = dict()
f = open('embeddings/glove.840B.300d/glove.840B.300d.txt',encoding='utf8')

for line in f:
    values = line.split(" ")
    words = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[words]= coefs
    
f.close()
embedding_matrix = np.zeros((max_features, 300))
for word, index in tokenizer.word_index.items():
    if index > max_features - 1:
        break
    else:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
print("Done")



# model
'''
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim= embed_size , input_length=maxlen,weights=[embedding_matrix], trainable=False))
model.add(Conv1D(64,3,strides=2,padding='same',activation='relu'))
model.add(Bidirectional(GRU(128,activation='relu',dropout=0.25,recurrent_dropout=0.25)))
model.add(Dropout(0.45))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
'''
model=Sequential()
model.add(Embedding(max_features, embed_size, weights=[embedding_matrix],input_length=maxlen,trainable = False))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
model.add(GlobalMaxPool1D())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Done")




#training
model.fit(train_X,train_y,epochs=2,batch_size=1024)
print("Done")

pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)

