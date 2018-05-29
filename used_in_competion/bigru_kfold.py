
# coding: utf-8

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk


# In[5]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
embeding_file_path = "glove.840B.300d.txt"


# In[6]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
wl = WordNetLemmatizer()

filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\''

def normalize(text):
    text = text.lower()
    translate_map = str.maketrans(filters, " " * len(filters))
    text = text.translate(translate_map)
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    stop_words = set(stopwords.words('english'))
    seq= [wl.lemmatize(t[0], pos=get_wordnet_pos(t[1])) for t in tags if t[0] not in stop_words]
    #seq= [wl.lemmatize(t[0], pos=get_wordnet_pos(t[1])) for t in tags]

    return seq
#s = normalize(train["comment_text"][3/


# In[7]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_features = 100000
embed_size = 300
maxlen = 150

data = pd.concat((train['comment_text'], test['comment_text']))
seqs = [normalize(text) for text in data]

def seq_to_sequence(seq, word_index):
    sequence = []
    for word in seq:
        if not word_index.get(word): continue 
        sequence.append(word_index[word])
    return sequence

def fit_on_sequence(seqs):
    word_counts = dict()
    for seq in seqs:
        for w in seq:
            if w not in word_counts:
                word_counts[w] = 0
            word_counts[w] += 1
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts if wc[1]>=10]
    sorted_voc = sorted_voc[:max_features]
    word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
    return word_index
    
word_index = fit_on_sequence(seqs)
train_words = [seq_to_sequence(seq, word_index) for seq in seqs[:train.shape[0]]]
test_words = [seq_to_sequence(seq, word_index) for seq in seqs[train.shape[0]:]]
train_words = pad_sequences(train_words, maxlen=maxlen )
test_words = pad_sequences(test_words, maxlen=maxlen)


# In[8]:


def get_coef(word, *coefs):
    return word, np.asarray(coefs, dtype=np.float32)
embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path)) 


# In[34]:


max_words = min(max_features, len(word_index))
embeding_matrix = np.zeros((max_words+1, embed_size))
lose = 0
lost_words = []
for word,i in word_index.items():
    if word not in embeding_dict: 
        lose += 1
        print(word)
        continue
    if i>max_words: 
        continue 
    embeding_matrix[i] = embeding_dict[word]
print(lose)


# In[35]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Conv1D, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,Dropout
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

filepath="weights_base.best.hdf5"
num_filters = 64 
filter_sizes = [2,3,4]
batch_size = 128
epochs = 4
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
x_train = train_words 
x_test = test_words

def get_model():
    inp = Input(shape=(maxlen,)) #maxlen
    x = Embedding(max_words+1, embed_size, weights=[embeding_matrix], trainable = False )(inp)
    x = SpatialDropout1D(0.2)(x)
    x,s_h,sc = Bidirectional(GRU(128, W_regularizer=None,return_sequences=True, return_state=True, dropout=0.1,recurrent_dropout=0.1))(x)
    #x,s_h,s_c = Bidirectional(GRU(80, return_sequences=True, return_state=True))(x)
    x = Conv1D(num_filters, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    pool = concatenate([avg_pool, max_pool])
    #h = Dropout(0.5)(pool)
    #h = Dense(40, activation="tanh" ,W_regularizer=l2(1e-1))(h)
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
            
def train_and_validate_model(x_tra, y_tra, x_val, y_val):
    model = get_model()
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 2)
    RocAuc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
    history = model.fit(x_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                 callbacks=[RocAuc,early_stop, checkpoint], verbose=1)
    return model


# In[36]:


from sklearn.model_selection import StratifiedKFold
nfolds = 10

skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
models = {}
y_preds = {}
for i, (tra, val) in enumerate(skf.split(x_train, train['toxic'])):
    print ("Running Fold", i+1, "/", nfolds)
    
    model = train_and_validate_model(x_train[tra], y_train[tra], x_train[val], y_train[val])
   # model.save("bigru_kfold_v{}.h5".format(i))
    model.load_weights(filepath)
    y_preds[i] = model.predict(x_test, batch_size=1024)
    
    break


# In[37]:


y_pred = np.zeros((x_test.shape[0], 6))
for i in range(nfolds):
    print(i)
    y_pred += y_preds[i]
    break
y_pred /= 1.0 #float(nfolds)
submission = pd.read_csv('sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('bigru_conv1d_bestmodel.csv', index=False)           
submission

