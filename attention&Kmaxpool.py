
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
embeding_file_path = "glove.840B.300d.txt"
embeding_file_path_1 = "crawl-300d-2M.vec"
test = test.fillna("NAN")
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))


# In[3]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_features = 150000
embed_size = 300
maxlen = 200

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(pd.concat((train['comment_text'],test['comment_text'])))
train_words = tokenizer.texts_to_sequences(train['comment_text'])
test_words = tokenizer.texts_to_sequences(test['comment_text'])
train_words = pad_sequences(train_words, maxlen=maxlen)
test_words = pad_sequences(test_words, maxlen=maxlen)


# In[4]:


import gc
def get_coef(word, *coefs):
    return word, np.asarray(coefs, dtype=np.float32)
embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path))
word_index = tokenizer.word_index
max_words = min(max_features, len(word_index))
embeding_matrix = np.zeros((max_words+1, embed_size))
lose = 0
lost_words = []
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


# In[5]:


def get_coef(word, *coefs):
    return word, np.asarray(coefs, dtype=np.float32)
embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path_1))
word_index = tokenizer.word_index
max_words = min(max_features, len(word_index))
embeding_matrix_1 = np.zeros((max_words+1, embed_size))
lose = 0
lost_words = []
for word,i in word_index.items():
    if word not in embeding_dict: 
        lose += 1
        word = "something"
    if i>max_words: 
        continue 
    embeding_matrix_1[i] = embeding_dict[word]
print(lose)
del embeding_dict
gc.collect()


# In[6]:


embeding_matrix = np.concatenate((embeding_matrix, embeding_matrix_1), axis=1)
del embeding_matrix_1
gc.collect()


# In[39]:


from keras.engine import Layer, InputSpec
from keras.layers import Flatten
import tensorflow as tf

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        
        # return flattened output
        return Flatten()(top_k)


# In[40]:


algo="textRNN_v2_kmaxpool"
local_cv_score = 0 

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate,merge
from keras.layers import CuDNNGRU, CuDNNLSTM,GRU, Conv1D, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,Dropout,Flatten,                 Activation,Reshape,RepeatVector,Permute,MaxPooling1D,Lambda,BatchNormalization as bn, TimeDistributed, Dot,Multiply
from keras.activations import softmax
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from keras import backend as K

model_filepath="{}.bestmodel.hdf5"
num_filters = 64 
filter_sizes = [2,3,4]
batch_size = 128
units = 128
epochs = 50
early_stop_rounds = 5 
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
x_train = train_words 
x_test = test_words

def get_model():
    inp = Input(shape=(maxlen,)) #maxlen
    x = Embedding(max_words+1, embed_size*2, weights=[embeding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(units,return_sequences=True))(x)
    x_1 = Bidirectional(CuDNNGRU(units,return_sequences=True))(x)
    # x shape is (batch_size, seqsize, units*2)
    
    ###### simple pooling layer 
#     max_pool = GlobalMaxPooling1D()(x)
#     ave_pool = GlobalAveragePooling1D()(x)
#     max_pool_1 = GlobalMaxPooling1D()(x_1)
#     ave_pool_1 = GlobalAveragePooling1D()(x_1)
#     pool = concatenate([max_pool, ave_pool, max_pool_1, ave_pool_1])
#     # pool shape is (batch_size, units*4)
#     pool = Dropout(0.3)(pool)


    ###### attention implmention att_w = softmax(tanh(h*w+b))
#     t = TimeDistributed(Dense(units, activation="tanh"))(x_1)
#     t = Lambda(lambda t: K.sum(t, axis=2), output_shape=(maxlen,))(t)# t shape is (batch_size, seqsize)
#     s = Activation("softmax")(t) # s shape is (batch_size, seqsize)
#     x_1 = Reshape((units*2, maxlen))(x_1)
#     aw = Multiply()([s,x_1]) # s shape is (batch_size, units*2, seqsize)
#     p =  Lambda(lambda s: K.sum(s, axis=2),output_shape=(units*2,))(aw)# s shape is (batch_size, units*2)
    
    ###### k-max pooling 
    x = KMaxPooling(3)(x)
    x_1 = KMaxPooling(3)(x_1) # x & x_1 shape is (batch_size, units*2*3)
    p = concatenate([x,x_1])
    p = Dropout(0.3)(p)
    
    z = Dense(1000)(p)
    z = bn()(z)
    z = Activation("relu")(z)
    z = Dropout(0.3)(z)
    
    z = Dense(400)(p)
    z = bn()(z)
    z = Activation("relu")(z)
    z = Dropout(0.3)(z)
    
    oup = Dense(6, activation='sigmoid',W_regularizer=None)(z)
    
    model = Model(input=inp, output=oup)
    model.compile(loss='binary_crossentropy',optimizer = Adam(lr = 1e-3, decay = 0.0), metrics=['accuracy'])
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
                self.model.save_weights(model_filepath.format(algo))
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


# In[41]:


from sklearn.model_selection import StratifiedKFold
nfolds = 10

skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
models = {}
y_preds = {}
shit_2 = np.zeros((train.shape[0], 6))

for i, (tra, val) in enumerate(skf.split(x_train, train['toxic'])):
    print ("Running Fold", i+1, "/", nfolds)
    
    model = train_and_validate_model(x_train[tra], y_train[tra], x_train[val], y_train[val])
    model.load_weights(model_filepath.format(algo))
    
    val_preds = model.predict(x_train[val], batch_size=1024)
    shit_2[val,:] = val_preds
    
    y_preds[i] = model.predict(x_test, batch_size=1024)

shit_2 = pd.DataFrame(shit_2)
shit_2.to_csv("{}_oofs.csv".format(algo))
y_pred = np.zeros((x_test.shape[0], 6))
for i in range(nfolds):
    print(i)
    y_pred += y_preds[i]
y_pred /= float(nfolds)

submission = pd.read_csv('sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('{}.csv'.format(algo), index=False)


# In[42]:


local_cv_score / 10.0

