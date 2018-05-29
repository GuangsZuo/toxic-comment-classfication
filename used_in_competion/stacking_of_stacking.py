
# coding: utf-8

# In[1]:


import xgboost as xgb
import pandas as pd 
import numpy as np


# In[15]:


param = {}
param["n_jobs"] = 10 
param['n_estimators'] = 500
param['objective'] = 'binary:logistic'
param['learning_rate'] = 0.095
param['max_depth'] = 5
param['silent'] = True
param['eval_metric'] = 'auc'
param['min_child_weight'] = 2
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7

numrounds = 500


# In[3]:


# res14 = pd.read_csv("glove_nn_bestmodel_10foldave.csv") #9494
# res15 = pd.read_csv("2layer_textcnn_bestmodel_10foldave.csv") #9829
# res16 = pd.read_csv("charcnn_bestmodel_10foldave.csv") #9776
# res17 = pd.read_csv("bigru_attention_bestmodel_10foldave.csv") #9848


# In[4]:


textcnn_oofs = pd.read_csv("textcnn_oofs.csv")
bigru_conv1d_oofs = pd.read_csv("bigru_conv1d_oofs.csv")
lr_oofs = pd.read_csv("tuned_lr_oofs.csv")
bigru2_oofs = pd.read_csv("2bigru_oofs.csv")
xgb_oofs = pd.read_csv("xgb_oofs.csv")
bigru_att_oofs = pd.read_csv("bigru_attention_oofs.csv")
charcnn_oofs = pd.read_csv("charcnn_oofs.csv")
#glove_nn_oofs = pd.read_csv("glove_nn.csv")
textcnn2_oofs = pd.read_csv("2layers_textcnn_oofs.csv")
fm_oofs = pd.read_csv("wordbatch_oof.csv")
nbsvm_oofs = pd.read_csv("nbsvm_oofs.csv")
#319
lgbm_oofs = pd.read_csv("lgbm_oofs.csv")
rf_oofs = pd.read_csv("rf_offs_v2.csv")
bigru2_rank_oofs = pd.read_csv("2bigru_rankave_oofs.csv")
bigru_conv_rank_oofs = pd.read_csv("bigru_conv_rankave_oofs.csv")
xgb_lr_oofs=pd.read_csv("xgb_lr_oofs.csv")
bigru2_prepro_oofs = pd.read_csv("2bigru_prepro_oofs.csv")
#320
capsule_oofs = pd.read_csv("capsuleNet_prepro_oofs.csv")
bigru_conv_prepro_nocu_oofs = pd.read_csv("bigru_conv_prepro_nocu_oofs.csv")
bigru2_prepro_nocu_oofs = pd.read_csv("2bigru_prepro_nocu_oofs.csv")
xgb_stacking_oofs = pd.read_csv("xgb_stacking_oofs.csv")
lgb_stacking_oofs = pd.read_csv("lgb_stacking_oofs.csv")
nn_stacking_oofs = pd.read_csv("nn_stacking_oofs.csv")

textcnn_subs = pd.read_csv("textcnn_bestmodel.csv")
bigru_conv1d_subs = pd.read_csv("bigru_conv1d_bestmodel_10foldsave.csv")
lr_subs = pd.read_csv("tuned_lr_10foldave.csv")
bigru2_subs = pd.read_csv("2bigru_bestmodel_10foldave.csv")
xgb_subs = pd.read_csv("xgb_tfidf_10foldave.csv")
bigru_att_subs = pd.read_csv("bigru_attention_bestmodel_10foldave.csv")
charcnn_subs = pd.read_csv("charcnn_bestmodel_10foldave.csv")
#glove_nn_subs = pd.read_csv("glove_nn_bestmodel_10foldave.csv")
textcnn2_subs = pd.read_csv("2layer_textcnn_bestmodel_10foldave.csv")
fm_subs = pd.read_csv("wordbatch_sub.csv")
nbsvm_subs = pd.read_csv("nbsvm_10foldave.csv")
# 3.19 new added 
lgbm_subs = pd.read_csv("lgbm_subs.csv")
rf_subs = pd.read_csv("rf_10folds_ave_v2.csv")
bigru2_rank_subs = pd.read_csv("2bigru_bestmodel_10folds_rankave.csv")
bigru_conv_rank_subs = pd.read_csv("bigru_conv_10folds_rankave.csv")
xgb_lr_subs = pd.read_csv("xgb_lr_10foldave.csv")
#319 v2
bigru2_prepro_subs = pd.read_csv("2bigru_prepro_10foldsave.csv")
#320 
capsule_subs = pd.read_csv("capsuleNet_prepro_10foldave.csv")
bigru_conv_prepro_nocu_subs = pd.read_csv("bigru_conv_prepro_nocu_10foldsave.csv")
bigru2_prepro_nocu_subs = pd.read_csv("2bigru_prepro_nocu_10foldsave.csv")
xgb_stacking_subs =pd.read_csv("xgb_stacking_final.csv")
lgb_stacking_subs = pd.read_csv("lgb_stacking_final.csv")
nn_stacking_subs = pd.read_csv("nn_stacking_final.csv")

train = pd.read_csv("train_f.csv")
test = pd.read_csv("test_f.csv")


# In[16]:


from sklearn.metrics import log_loss,roc_auc_score
    
def train_and_validate_model(x_train, y_train, x_validation, y_validation, cls):
    clf = xgb.XGBClassifier(**param)
    clf.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_validation, y_validation)], eval_metric="auc",early_stopping_rounds=5, verbose=True)
    return clf


# In[17]:


lables = ['0', '1', '2', '3', '4', '5']
lables_1 = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
meta_features=["caps_vs_length","num_exclamation_marks","num_question_marks","num_punctuation","num_symbols",
               "num_words", "words_vs_unique","num_smilies"]
train_oofs = [ xgb_stacking_oofs[lables], lgb_stacking_oofs[lables], nn_stacking_oofs[lables],
              train[meta_features]]
test_subs = [xgb_stacking_subs[lables_1], lgb_stacking_subs[lables_1], nn_stacking_subs[lables_1],
             
             test[meta_features]]
x_train = pd.concat(train_oofs, axis=1).values
x_test =pd.concat(test_subs, axis=1).values


# In[18]:


from sklearn.model_selection import StratifiedKFold
nfolds = 10

skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
submission = pd.read_csv('sample_submission.csv')

y_pred = np.zeros((test.shape[0], 6))
y_preds = {}
shit_2 = np.zeros((train.shape[0], 6))

for index,cls in enumerate(train.columns[2:8]):
    print(cls)
    for i, (tra, val) in enumerate(skf.split(x_train, train[cls])):
        print ("Running Fold", i+1, "/", nfolds, "at class ", cls)
        model = train_and_validate_model(x_train[tra], train[cls][tra], x_train[val], train[cls][val], cls)
        y_pred[:,index] += model.predict_proba(x_test)[:, 1]
        shit_2[val,index] = model.predict_proba(x_train[val])[:,1]
y_pred /= float(nfolds)



submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('xgb_stacking_of_stacking_gl_v2.csv', index=False)


# In[12]:


test["comment_text"][1]


# In[21]:


import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score



def lgb_cv(min_child_weight, colsample_bytree, max_depth, subsample, learning_rate, reg_lambda, n_estimators ):
    params = {}
    params['max_depth'] = int(max_depth)
    params['learning_rate'] = float(learning_rate)
    params['n_estimators'] = int(n_estimators)
    params['objective'] = "binary"
    params['min_child_weight'] =float(min_child_weight)
    params['subsample'] = float(subsample)
    params['colsample_bytree'] = float(colsample_bytree)
    params['njobs']=20
    params['silent'] = True
    params['reg_lambda'] = float(reg_lambda)
    params['eval_metric']="auc"
    
    model = lgb.LGBMClassifier(**params)
    
    result = cross_val_score(model, x_train, train['toxic'], n_jobs=5, cv=5, scoring="roc_auc",verbose =1)
    return np.mean(result)

lgbBO = BayesianOptimization(lgb_cv, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.1, 1),
                                                'max_depth': (5, 15),
                                                'subsample': (0.5, 1),
                                                'learning_rate': (0, 1),
                                                'reg_lambda': (0, 1.0),
                                                'n_estimators':(20,200),
                                                })
lgbBO.explore({"min_child_weight":[2,5], "colsample_bytree":[0.7,0.8], "max_depth":[5,10], "learning_rate":[0.095,0.001],  
               'subsample':[0.7,0.6], "reg_lambda":[0,0.001],'n_estimators':[100,50]} )
lgbBO.maximize(init_points = 5, n_iter=20)


# In[8]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss,roc_auc_score
import lightgbm as lgb    
def train_and_validate_model(x_train, y_train, x_validation, y_validation, cls):
    params = {}
    params['max_depth'] = 5
    params['learning_rate'] = 0.1
    params['n_estimators'] = 500
    params['objective'] = "binary"
    params['min_child_weight'] = 15
    params['subsample'] = 0.6
    params['colsample_bytree'] =0.25
    params['njobs']=20
    params['silent'] = True
    params['reg_lambda'] = 0.8
    params['eval_metric']="auc"
    clf = lgb.LGBMClassifier(**params)
    clf.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_validation, y_validation)], eval_metric="auc",early_stopping_rounds=20, verbose=True)
    return clf,clf.best_iteration_


nfolds = 10
skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
submission = pd.read_csv('sample_submission.csv')

y_pred = np.zeros((test.shape[0], 6))
y_preds = {}
shit_2 = np.zeros((train.shape[0], 6))

for index,cls in enumerate(train.columns[2:8]):
    print(cls)
    for i, (tra, val) in enumerate(skf.split(x_train, train[cls])):
        print ("Running Fold", i+1, "/", nfolds, "at class ", cls)
        model, best_iter = train_and_validate_model(x_train[tra], train[cls][tra], x_train[val], train[cls][val], cls)
        y_pred[:,index] += model.predict_proba(x_test, num_iteration=best_iter)[:, 1]
        shit_2[val,index] = model.predict_proba(x_train[val])[:,1]
        
y_pred /= float(nfolds)
shit_2 = pd.DataFrame(shit_2)
shit_2.to_csv("lgb_stacking_oofs.csv")

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('lgb_stacking_final.csv', index=False)


# In[14]:


import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
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
from sklearn.model_selection import StratifiedKFold
nfolds = 3

skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
maxlen = x_train.shape[1]
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

def get_model(unit1,unit2,unit3,lamb, epoch):
    
    inp = Input(shape=(maxlen,)) #maxlen
    x = Dense(unit1, activation="relu", W_regularizer=l2(lamb))(inp)
    x = Dense(unit2, activation="relu", W_regularizer=l2(lamb))(x)
    x = Dense(unit3, activation="relu", W_regularizer=l2(lamb))(x)
    oup = Dense(6, activation='sigmoid',W_regularizer=None)(x)
    
    model = Model(input=inp, output=oup)
    model.compile(loss='binary_crossentropy',optimizer = Adam(lr = 1e-3, decay = 0.0), metrics=['accuracy'])
    return model

def nn_cv(unit1, unit2, unit3, lamb, epoch):
    unit1 = int(unit1)
    lamb = float(lamb)
    unit2 = int(unit2)
    unit3 = int(unit3)
    epoch = int(epoch)
    
    model = get_model(unit1,unit2,unit3,lamb, epoch)
    
    score = 0 
    for i, (tra, val) in enumerate(skf.split(x_train, train['toxic'])):
        print ("Running Fold", i+1, "/", nfolds)
        model.fit(x_train[tra], y_train[tra], batch_size=128, epochs=epoch)

        val_preds = model.predict(x_train[val], batch_size=1024)
        score += roc_auc_score(y_train[val], val_preds)
    score /= float(nfolds)
    return score

lgbBO = BayesianOptimization(nn_cv, {"unit1":(30,200),"unit2":(30,200), 'unit3':(10,100), "lamb":(0,0.1),"epoch":(3,8)
                                               
                                
                                                })
lgbBO.explore({"unit1":[30,50,100],"unit2":[30,50,100], 'unit3':[30,50,100], "lamb":[0,0.001,0.1], "epoch":[3,4,5]} )
lgbBO.maximize(init_points = 5, n_iter=20)


# In[10]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss,roc_auc_score
import lightgbm as lgb    
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
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
maxlen = x_train.shape[1]
filepath = "nn-stacking-{}.hdf5"
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values


def get_model(unit1,unit2,unit3,lamb, epoch):
    inp = Input(shape=(maxlen,)) #maxlen
    x = Dense(unit1, activation="relu", W_regularizer=l2(lamb))(inp)
    x = Dense(unit2, activation="relu", W_regularizer=l2(lamb))(x)
    x = Dense(unit3, activation="relu", W_regularizer=l2(lamb))(x)
    oup = Dense(6, activation='sigmoid',W_regularizer=None)(x)
    
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
    model = get_model(100,200,20,0,5)
    checkpoint = ModelCheckpoint(filepath.format(ith), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 4)
    RocAuc = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)
    history = model.fit(x_tra, y_tra, batch_size=128, epochs=5, validation_data=(x_val, y_val),
                 callbacks=[RocAuc,early_stop, checkpoint], verbose=1)
    return model


nfolds = 10
skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
submission = pd.read_csv('sample_submission.csv')

y_pred = np.zeros((test.shape[0], 6))
y_preds = {}
shit_2 = np.zeros((train.shape[0], 6))

for i, (tra, val) in enumerate(skf.split(x_train, train["toxic"])):
    print ("Running Fold", i+1, "/", nfolds, "at class ", cls)
    model = train_and_validate_model(x_train[tra], y_train[tra], x_train[val], y_train[val], i)
    model.load_weights(filepath.format(i))
    y_pred += model.predict(x_test, batch_size=1024)
    shit_2[val,:] = model.predict(x_train[val], batch_size=1024)
    
y_pred /= float(nfolds)
shit_2 = pd.DataFrame(shit_2)
shit_2.to_csv("nn_stacking_oofs.csv")

submission = pd.read_csv('sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('nn_stacking_final.csv', index=False)

