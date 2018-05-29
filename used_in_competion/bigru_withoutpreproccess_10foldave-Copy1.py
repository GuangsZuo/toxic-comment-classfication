
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
embeding_file_path = "glove.840B.300d.txt"
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"


# In[3]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_features = 200000
embed_size = 300
maxlen = 300

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(pd.concat((train['comment_text'],test['comment_text'])))
train_words = tokenizer.texts_to_sequences(train['comment_text'])
test_words = tokenizer.texts_to_sequences(test['comment_text'])
train_words = pad_sequences(train_words, maxlen=maxlen)
test_words = pad_sequences(test_words, maxlen=maxlen)


# In[4]:


def get_coef(word, *coefs):
    return word, np.asarray(coefs, dtype=np.float32)
embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path)) 


# In[5]:


# #fast-text Enriching Word Vectors with Subword Information
# from gensim.models import FastText
# sentences = seqs 
# word_vec = FastText(sentences, hs=1, min_count=3, size=300,sg=1)
# word_vec.save("gensim_fasttext")


# In[6]:


word_index = tokenizer.word_index
max_words = min(max_features, len(word_index))
embeding_matrix = np.zeros((max_words+1, embed_size))
lose = 0
lost_words = []
for word,i in word_index.items():
    if word not in embeding_dict: 
        lose += 1
        continue
    if i>max_words: 
        continue 
    embeding_matrix[i] = embeding_dict[word]
print(lose)


# In[17]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate,merge
from keras.layers import CuDNNGRU,GRU, Conv1D, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,Dropout,Flatten,                 Activation,Reshape,RepeatVector,Permute,MaxPooling1D,Lambda
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from keras import backend as K

filepath="bigru_conv_rankave-{}.best.hdf5"
num_filters = 100 
filter_sizes = [2,3,4]
batch_size = 128
epochs = 4
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
x_train = train_words 
x_test = test_words

def get_model():
    inp = Input(shape=(maxlen,)) #maxlen
    x = Embedding(max_words+1, embed_size, weights=[embeding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.2)(x)
    #x,s_h,sc = Bidirectional(CuDNNGRU(128,return_sequences=True, return_state=True))(x)
    #x = Dropout(0.3)(x)
#     x_1,s_h_1,sc = Bidirectional(CuDNNGRU(128,return_sequences=True, return_state=True))(x)
#     x_1 = Dropout(0.3)(x_1)
    x,s_h,s_c = Bidirectional(GRU(128, return_sequences=True, return_state=True,dropout=0.1,recurrent_dropout=0.2))(x)
    x,s_h,s_c = Bidirectional(GRU(128, return_sequences=True, return_state=True,dropout=0.1,recurrent_dropout=0.2))(x)
    x = Conv1D(num_filters, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
#     avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
   # max_pool_1 =  GlobalMaxPooling1D()(x_1)
#     pool = concatenate([avg_pool, max_pool, s_h])
    #h = Dropout(0.5)(pool)
    #h = Dense(40, activation="tanh" ,W_regularizer=l2(1e-1))(h)
    
#     h = Dense(128, activation="tanh")(x)
#     s = Dense(1)(h)
#     s = Reshape((maxlen,))(s)
#     s = Activation("softmax")(s)
#     s = RepeatVector(128*2)(s)
#     s = Permute((2,1))(s)
#     attention = merge([x, s], mode='mul')
#     myreduce = Lambda(lambda x: K.sum(x, axis=1))
#     attention = myreduce(attention)

    pool = concatenate([max_pool,  s_h])
    pool = Dropout(0.1)(pool)
    oup = Dense(6, activation='sigmoid',W_regularizer=None)(pool)
    
    model = Model(input=inp, output=oup)
    model.compile(loss='binary_crossentropy',optimizer = Adam(lr = 1e-3, decay = 0.0), metrics=['accuracy'])
    return model

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            
def train_and_validate_model(x_tra, y_tra, x_val, y_val, ith):
    model = get_model()
    checkpoint = ModelCheckpoint(filepath.format(ith), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2)
    RocAuc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
    history = model.fit(x_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                 callbacks=[RocAuc,early_stop], verbose=1)
    return model


# In[18]:


from sklearn.model_selection import StratifiedKFold
nfolds = 10

skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
models = {}
y_preds = {}
shit_2 = np.zeros((train.shape[0], 6))

for i, (tra, val) in enumerate(skf.split(x_train, train['toxic'])):
    print ("Running Fold", i+1, "/", nfolds)
    
    model = train_and_validate_model(x_train[tra], y_train[tra], x_train[val], y_train[val], i)
   # model.save("bigru_kfold_v{}.h5".format(i))
    #model.load_weights(filepath.format(i))
    
    val_preds = model.predict(x_train[val], batch_size=1024)
    shit_2[val,:] = val_preds
    
    y_preds[i] = model.predict(x_test, batch_size=1024)



# In[19]:


shit_2 = pd.DataFrame(shit_2)
shit_2.to_csv("bigru_conv_rankave_oofs.csv")


# In[22]:


y_pred = np.zeros((x_test.shape[0], 6))
for i in range(nfolds):
    print(i)
    y_pred += y_preds[i]
y_pred /= float(nfolds)
# from scipy.stats import rankdata
# for f in range(nfolds):
#     for i in range(6):
#         y_pred[:,i] =  np.add(y_pred[:,i], rankdata(y_preds[f][:,i])/float(y_pred.shape[0]))
# y_pred /= float(nfolds)

submission = pd.read_csv('sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('bigru_conv_10folds_ave_v2.csv', index=False)
#submission.to_csv('bigru_attention_bestmodel_10foldave.csv', index=False)
submission  

