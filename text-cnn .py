
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
embeding_file_path = "glove.840B.300d.txt"
embeding_file_path_1 = "crawl-300d-2M.vec"
test=test.fillna("NAN")


# In[3]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_features = 100000
embed_size = 300
maxlen = 150

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(pd.concat((train['comment_text'],test['comment_text'])))
train_words = tokenizer.texts_to_sequences(train['comment_text'])
test_words = tokenizer.texts_to_sequences(test['comment_text'])
train_words = pad_sequences(train_words, maxlen=maxlen)
test_words = pad_sequences(test_words, maxlen=maxlen)


# In[4]:


train_words.shape


# In[5]:


def get_coef(word, *coefs):
    return word, np.asarray(coefs, dtype=np.float32)
embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path))
word_index = tokenizer.word_index
max_words = min(max_features, len(word_index))
embeding_matrix = np.random.randn(max_words+1, embed_size)
lose =0 
for word,i in word_index.items():
    if word not in embeding_dict: 
        lose += 1
        word = "something"
    if i>max_words:
        continue
    embeding_matrix[i] = embeding_dict[word]
print(lose)
del embeding_dict
gc.collect()


# In[ ]:


embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path_1))
embeding_matrix_1 = np.random.randn(max_words+1, embed_size)
lose =0 
for word,i in word_index.items():
    if word not in embeding_dict: 
        lose += 1
        word = "something"
    if i>max_words:
        continue
    embeding_matrix_1[i] = embeding_dict[word]
del embeding_dict
gc.collect()
print(lose)


# In[6]:


embeding_matrix = np.load("embeding-600d.npy")


# In[9]:


algo="3layerCNN_v7"
local_cv_score = 0 

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Reshape, Flatten, Dropout, Conv2D, MaxPool2D,GlobalMaxPooling2D,BatchNormalization as bn,Activation
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from sklearn.metrics import roc_auc_score
from keras import regularizers

num_filters = 100
filter_sizes = [2,3,4,5]
model_filepath="./modelsave/{}.bestmodel.hdf5".format(algo)
batch_size = 128
epochs = 50
early_stop_rounds = 5

def get_model():    
    inp = Input(shape=(maxlen, ))
    x = Embedding(embeding_matrix.shape[0], embed_size*2, weights=[embeding_matrix], trainable=False)(inp)
    #x_1 = Embedding(max_words+1, embed_size, weights=[embeding_matrix_1], trainable=False)(inp)
    # (batch_size, seqsize, embed_size)
    x = SpatialDropout1D(0.1)(x)
    #x_1 = SpatialDropout1D(0.1)(x_1)
    x = Reshape((maxlen, embed_size*2, 1))(x)
    #x_1 = Reshape((maxlen, embed_size, 1))(x_1)
    #x = concatenate([x,x_1],axis=3)
    
    ys = []
    # (batch_size, len, embed_size, channel)
    for filter_size in filter_sizes:
        conv = Conv2D(num_filters, kernel_size=(filter_size, embed_size*2), kernel_initializer="normal")(x)
        # conv output-> (batch_size, len-filter_size+1, 1, num_filters)
        bnlayer = bn()(conv)
        relu = Activation("relu")(bnlayer)
        conv = Conv2D(num_filters, kernel_size= (filter_size, 1), kernel_initializer="normal")(relu)
        # conv output -> (batch_size, len-2*filter_size+1, 1, num_filters)
        bnlayer = bn()(conv)
        relu = Activation("relu")(bnlayer)
        
        # comment for not improve local_cv 
#         conv = Conv2D(num_filters, kernel_size= (filter_size, 1), kernel_initializer="normal")(relu)
#         # conv output -> (batch_size, len-2*filter_size+1, 1, num_filters)
#         bnlayer = bn()(conv)
#         relu = Activation("relu")(bnlayer)

        maxpool = GlobalMaxPooling2D()(relu)
        # maxpool shape -> (batch_size, num_filters)
        ys.append(maxpool)
    
    z = Concatenate(axis=1)(ys) 
    # z shape -> (batch_size, num_filters*4)
    z = Dropout(0.2)(z)
    z = Dense(300)(z)
    z = bn()(z)
    z = Activation("relu")(z)
    z = Dropout(0.2)(z)
    
    z = Dense(200)(z)
    z = bn()(z)
    z = Activation("relu")(z)
    z = Dropout(0.2)(z)
    outp = Dense(6, activation="sigmoid" )(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
    
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        global local_cv_score
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            if (score > self.max_score):
                print("*** New High Score (previous: %.6f) \n" % self.max_score)
                self.model.save_weights(model_filepath)
                self.max_score=score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > early_stop_rounds:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.max_score))
                    self.model.stop_training = True
                    local_cv_score += self.max_score 
        
def train_and_validate_model(x_tra, y_tra, x_val, y_val):
    model = get_model()
    RocAuc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
#     board=TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, 
#                                write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    history = model.fit(x_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                 callbacks=[RocAuc], verbose=1)
    return model


# In[10]:


from sklearn.model_selection import StratifiedKFold
nfolds = 10


y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
x_train = train_words 
x_test = test_words
skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
models = {}
y_preds = {}
shit = np.zeros((train.shape[0], 6))

for i, (tra, val) in enumerate(skf.split(x_train, train['toxic'])):
    print ("Running Fold", i+1, "/", nfolds)
    
    model = train_and_validate_model(x_train[tra], y_train[tra], x_train[val], y_train[val])
    model.load_weights(model_filepath)
    
    val_preds = model.predict(x_train[val], batch_size=1024)
    shit[val,:] = val_preds
    
    y_preds[i] = model.predict(x_test, batch_size=1024)
    
shit = pd.DataFrame(shit)
shit.to_csv("./result/{}_oofs.csv".format(algo))    
y_pred = np.zeros((x_test.shape[0], 6))
for i in range(nfolds):
    print(i)
    y_pred += y_preds[i]
y_pred /= float(nfolds)
submission = pd.read_csv('sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('./result/{}.csv'.format(algo), index=False)           


# In[11]:


local_cv_score / 10

