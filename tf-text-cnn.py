
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
embeding_file_path = "glove.840B.300d.txt"
embeding_file_path_1 = "crawl-300d-2M.vec"
test=test.fillna("NAN")


# In[3]:


max_features = 100000
embed_size = 300
maxlen = 150


# In[4]:


newer = 0
if newer:
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(pd.concat((train['comment_text'],test['comment_text'])))
    train_words = tokenizer.texts_to_sequences(train['comment_text'])
    test_words = tokenizer.texts_to_sequences(test['comment_text'])
    train_words = pad_sequences(train_words, maxlen=maxlen)
    test_words = pad_sequences(test_words, maxlen=maxlen)
    
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

    embeding_matrix = np.concatenate((embeding_matrix, embeding_matrix_1), axis=1)
    del embeding_matrix_1
    gc.collect()
    
    np.save("embeding-600d",embeding_matrix)
    np.save("train_words",train_words)
    np.save("test_words",test_words)
else:
    embeding_matrix = np.load("embeding-600d.npy")
    train_words = np.load("train_words.npy")
    test_words = np.load("test_words.npy")


# In[5]:


test_words.shape


# In[6]:


import tensorflow as tf


tf.reset_default_graph()  # clear graph

batch_size = 128
# filters = 100
# filter_sizes = [1,2,3,4,5]
# strides = 1

units = 128

inp = tf.placeholder(tf.int32, shape=(None, maxlen),name="input")
target = tf.placeholder(tf.float32, shape=(None, 6),name="class_label")
word_embeding = tf.get_variable("embeding", shape=embeding_matrix.shape, initializer = tf.constant_initializer(embeding_matrix))

feature = tf.nn.embedding_lookup(word_embeding, inp) # (batch_size, seqsize, embed_size*2)
feature = tf.reshape(feature, (-1,maxlen, embed_size*2, 1))
pool_layers = [] 
for filter_size in filter_sizes:
    with tf.name_scope("gru-layer"):
#         filter_shape = [filter_size, embed_size*2, 1, filters]
#         w_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter")
#         conv_layer = tf.nn.conv2d(feature, w_filter, strides=(1,1,1,1),padding="VALID")
        conv_layer = tf.layers.conv2d(feature, filters, (filter_size,embed_size*2), strides=(1,1),
                                     padding='valid', kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)) 
        norm_layer = tf.layers.batch_normalization(conv_layer)
        relu_layer = tf.nn.relu(norm_layer) # (batch_size, seqsize-filtersize+1,1,filters)
        #reshape_layer = tf.reshape(relu_layer, (-1, maxlen-filter_size+1, filters, 1))
    with tf.name_scope("convolution-layer-2"):
#         filter_shape = [filter_size, 1, filters, filters]
#         w_filter_1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter_1")
#         conv_layer = tf.nn.conv2d(relu_layer, w_filter_1, strides=(1,1,1,1),padding="VALID")
        conv_layer  = tf.layers.conv2d (relu_layer, filters, (filter_size,1), strides=(1,1),
                                     padding='valid', kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        norm_layer =tf.layers.batch_normalization(conv_layer)
        relu_layer = tf.nn.relu(norm_layer) # (batch_size, seqsize-filtersize*2+2 ,1,filters)
    with tf.name_scope("pooling-layer"):
        pool_size = maxlen - filter_size*2 + 2
        pool_layer = tf.layers.max_pooling2d(relu_layer, (pool_size,1), strides=(pool_size,1), padding="valid") # (batch_size, 1,1, filters)
    pool_layers.append(pool_layer)
    
fc_layer = tf.concat(pool_layers, axis=3)
fc_layer_flat = tf.reshape(fc_layer,(-1, len(filter_sizes)*filters)) # (batch_size, 300)

z = tf.layers.dense(fc_layer_flat,200, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
z = tf.layers.batch_normalization(z)
z = tf.nn.relu(z)
z = tf.nn.dropout(z, 0.8)

prob_logits = tf.layers.dense(z, 6, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),name="finallayer")
oup = tf.sigmoid(prob_logits)
losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=prob_logits) # (batch_size,)
loss = tf.reduce_mean(losses) # float

global_step = tf.get_variable("global_step", shape=(),dtype=tf.int32, trainable=False, initializer=tf.constant_initializer(1))
#lr = tf.get_variable("learning_rate", shape=(),dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(1e-3))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    trainop = tf.contrib.layers.optimize_loss(loss, global_step, 1e-3, 'Adam', summaries=["gradients","loss","learning_rate"])
#     tvars = tf.trainable_variables()
#     grads, _ = tf.gradients(loss, tvars)
#     optimizer1 = tf.train.AdamOptimizer(learning_rate=lr)
#     train_op = optimizer.apply_gradients(zip(grads1, tvars1),global_step=global_step)
merged = tf.summary.merge_all()


config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
writer = tf.summary.FileWriter("./tflogs/", sess.graph)
saver = tf.train.Saver()
algo = "tf-2layertextcnn"
modelpath = "./modelsave/{}.bestmodel.hdf5".format(algo)


# In[7]:


local_cv_score = 0
from sklearn.metrics import roc_auc_score
def get_val_loss(x_val, y_val, batch_size=256):
    loss = 0
    preds = np.zeros((x_val.shape[0],6))
    #import pdb;pdb.set_trace()
    for nstart, nend in zip(range(0, x_val.shape[0], batch_size), range(batch_size, x_val.shape[0]+batch_size, batch_size)):
        if nend > x_val.shape[0]: nend = x_val.shape[0]
        x_val_batch = x_val[nstart:nend]
        y_val_batch = y_val[nstart:nend]
        val_loss,preds_batch = sess.run([losses, oup], feed_dict={inp: x_val_batch, target: y_val_batch})
        loss += np.sum(val_loss) 
        preds[nstart:nend] = preds_batch
    return loss / len(x_val) / 6.0, preds

def model_train(x_train, y_train, x_val, y_val, epoch=100, early_stopping_rounds=5, batch_size=128):
    global local_cv_score   
    best_score = 0 
    bad_rounds = 0
    for epoch in range(1, epoch+1):
        train_loss = 0
        counter = 0
        #import pdb;pdb.set_trace()
        for nstart, nend in zip(range(0, x_train.shape[0], batch_size), range(batch_size, x_train.shape[0]+batch_size, batch_size)):
            counter += 1
            if nend > x_train.shape[0]: nend = x_train.shape[0]
            x_train_batch = x_train[nstart:nend]
            y_train_batch = y_train[nstart:nend]
            if counter % 10 == 0 :
                temp_loss,_,summary,step= sess.run([losses,trainop,merged,global_step], feed_dict={inp: x_train_batch, target: y_train_batch})
                train_loss += np.sum(temp_loss)
                writer.add_summary(summary,step)
            else:
                temp_loss,_= sess.run([losses,trainop], feed_dict={inp: x_train_batch, target: y_train_batch})
                train_loss += np.sum(temp_loss)
        train_loss = train_loss / len(x_train)/ 6.0
        val_loss,y_preds = get_val_loss(x_val, y_val, batch_size=1024)
        print("Epoch %d: train loss : %.6f, val loss: %.6f"%(epoch, train_loss, val_loss))
        score = roc_auc_score(y_val, y_preds)
        print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch, score))
        if score < best_score:
            bad_rounds += 1
            if bad_rounds >= early_stopping_rounds:
                print("Epoch %05d: early stopping, high score = %.6f" % (epoch,best_score))
                local_cv_score += best_score 
                break
        else:
            print("*** New High Score (previous: %.6f) \n" % best_score)
            saver.save(sess, modelpath)
            best_score = score 
            bad_rounds = 0
    
def predict(x_test, batch_size = 1024):
    result = np.zeros((x_test.shape[0], 6))
    saver.restore(sess, modelpath)
    for nstart, nend in zip(range(0, x_test.shape[0], batch_size), range(batch_size,x_test.shape[0]+batch_size,batch_size)):
        if nend>x_test.shape[0]: nend = x_test.shape[0]
        x_test_batch = x_test[nstart:nend]
        prob = sess.run(oup, feed_dict={inp: x_test_batch})
        result[nstart:nend] = prob
    return result


# In[8]:


from sklearn.model_selection import StratifiedKFold
kfolds = 10
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
x_train = train_words
x_test = test_words
skf = StratifiedKFold(n_splits=kfolds, shuffle=True)
models = {}
y_preds = {}

res = np.zeros((len(x_test),6))

for i, (tra, val) in enumerate(skf.split(x_train, train['toxic'])):
    print("Running Fold", i + 1, "/", kfolds)
    sess.run(tf.global_variables_initializer())
    model_train(x_train[tra], y_train[tra], x_train[val], y_train[val], epoch=100, early_stopping_rounds=5)
    y_preds[i] = predict(x_test, batch_size=1024)
    res += y_preds[i]
res /= 10.0
respd = pd.read_csv("sample_submission.csv")
respd[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = res
respd.to_csv("{}.csv".format(algo),index=False)


# In[10]:


local_cv_score / 10 

