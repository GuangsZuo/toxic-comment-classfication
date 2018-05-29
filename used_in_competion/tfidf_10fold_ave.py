
# coding: utf-8

# In[1]:


# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack, csr_matrix

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
embeding_file_path = "glove.840B.300d.txt"


# In[2]:


filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\''

def normalize(text):
    text = text.lower()
    translate_map = str.maketrans(filters, " " * len(filters))
    text = text.translate(translate_map)
    seq = text.split(" ")
    seq = [i for i in seq if i]
    return seq

data = pd.concat((train['comment_text'], test['comment_text']))
seqs = [normalize(text) for text in data]

def get_coef(word, *coefs):
    return word, np.asarray(coefs, dtype=np.float32)
embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path))

embed = np.zeros((len(seqs), 300))
for index,seq in enumerate(seqs):
    count = 0 
    for w in seq:
        if w in embeding_dict:
            embed[index,:] += embeding_dict[w]
            count+=1
    if count > 0:
        embed[index,:] /= float(count)


# In[3]:


np.argwhere(np.isnan(embed))


# In[4]:



document = pd.concat((train['comment_text'],test['comment_text']))
document.fillna('')

tfidf_1gram = tfidf(stop_words="english", ngram_range=(1,4), max_features=50000, sublinear_tf=True, strip_accents="unicode", min_df=3, max_df=0.9)
#tfidf_2gram = tfidf(stop_words="english", ngram_range=(2,4), max_features=20000, sublinear_tf=True, strip_accents="unicode", min_df=3)
#tfidf_chargram = tfidf(encoding='unicode', analyzer='char', ngram_range=(2,6), sublinear_tf=True, max_features=40000)

tfidf_1gram = tfidf_1gram.fit(document)
#tfidf_2gram = tfidf_2gram.fit(document)
#tfidf_chargram = tfidf_chargram.fit(document)
train_f= pd.read_csv("train_f.csv")
test_f = pd.read_csv("test_f.csv")

train_tfidf = tfidf_1gram.transform(train['comment_text'])
test_tfidf = tfidf_1gram.transform(test['comment_text'])


# In[10]:


hand_features=["caps_vs_length", "words_vs_unique",]

# train_data = hstack((train_tfidf, csr_matrix(embed[:train.shape[0]])))

# test_data = hstack((test_tfidf, csr_matrix(embed[train.shape[0]:])))

train_data = train_tfidf
test_data = test_tfidf


# In[6]:


np.argwhere(np.isnan(test_f[hand_features].values))


# In[7]:


from scipy import sparse as sp
sp.save_npz("test_data_featured.npz", test_data)
sp.save_npz("train_data_featured.npz", train_data)


# In[2]:


from scipy import sparse as sp
test_data = sp.load_npz("test_data_featured.npz")
train_data = sp.load_npz("train_data_featured.npz")



# In[3]:


from sklearn.metrics import log_loss,roc_auc_score


def train_and_validate_model(model, x_train, y_train, x_validation, y_validation, cls):
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_validation)[:, 1]
    y_pred_tra = model.predict_proba(x_train)[:, 1]
    roc_score = roc_auc_score(y_validation, y_pred)
    val_loss_score = log_loss(y_validation, y_pred)
    train_loss_score = log_loss(y_train, y_pred_tra)
    print("logistic regression: train_loss: {} . val_loss: {}. roc_auc: {}".format(train_loss_score, val_loss_score, roc_score))
    return model


# In[11]:


from sklearn.model_selection import StratifiedKFold
#from scipy.sparse import coo_matrix
nfolds =10

skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
submission = pd.read_csv('sample_submission.csv')
y_pred = np.zeros((test.shape[0], 6))

x_train = train_data.tocsr()   # csr is ok to index 
x_test = test_data

lr_oofs = np.zeros((train.shape[0],6))
for index,cls in enumerate(train.columns[2:]):
    print(cls)
    for i, (tra, val) in enumerate(skf.split(x_train, train[cls])):
        print ("Running Fold", i+1, "/", nfolds, "at class ", cls)
        lr = LogisticRegression(solver='sag')
        model = train_and_validate_model(lr, x_train[tra], train[cls][tra], x_train[val], train[cls][val], cls)
        y_pred[:,index] += model.predict_proba(x_test)[:, 1]
        
        lr_oofs[val,index] = model.predict_proba(x_train[val])[:, 1]
        
y_pred /= float(nfolds)
lr_oofs = pd.DataFrame(lr_oofs)
lr_oofs.to_csv("lr_offs_v2.csv")

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('lr_10folds_ave_v2.csv', index=False)


# In[4]:


submission = pd.DataFrame()
submission['id'] = test['id']
for nclass in train.columns[2:]:
    lr = LogisticRegression(solver='sag')
    lr.fit(train_data, train[nclass])
    submission[nclass]=lr.predict_proba(test_data)[:,1]
    cv_score = np.mean(cross_val_score(lr, train_data, train[nclass], cv=5, scoring='roc_auc'))
    print('the {} score is {}'.format(nclass, cv_score))

submission.to_csv("lr_5flods_ave.csv", index=False)


# In[2]:


from scipy import sparse as sp
test_data = sp.load_npz("test_data_featured.npz")
train_data = sp.load_npz("train_data_featured.npz")


# In[5]:


from sklearn.ensemble import RandomForestClassifier as rfc 
from sklearn.model_selection import StratifiedKFold
#from scipy.sparse import coo_matrix
nfolds =5

skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
submission = pd.read_csv('sample_submission.csv')
y_pred = np.zeros((test.shape[0], 6))

x_train = train_data.tocsr()   # csr is ok to index 
x_test = test_data

rf_oofs = np.zeros((train.shape[0],6))
for index,cls in enumerate(train.columns[2:]):
    print(cls)
    for i, (tra, val) in enumerate(skf.split(x_train, train[cls])):
        print ("Running Fold", i+1, "/", nfolds, "at class ", cls)
        rf = rfc(n_estimators=250, max_features=0.7, max_depth=5, n_jobs=10)
        model = train_and_validate_model(rf, x_train[tra], train[cls][tra], x_train[val], train[cls][val], cls)
        y_pred[:,index] += model.predict_proba(x_test)[:, 1]
        
        rf_oofs[val,index] = model.predict_proba(x_train[val])[:, 1]
        
y_pred /= float(nfolds)
rf_oofs = pd.DataFrame(rf_oofs)
rf_oofs.to_csv("rf_offs_v2.csv")

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('rf_10folds_ave_v2.csv', index=False)


# In[3]:


import xgboost as xgb
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

from sklearn.metrics import log_loss,roc_auc_score
    
def train_and_validate_model(x_train, y_train, x_validation, y_validation, cls):
    clf = xgb.XGBClassifier(**param)
    clf.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_validation, y_validation)], eval_metric="auc",early_stopping_rounds=10, verbose=True)
    return clf


# In[10]:


from sklearn.model_selection import StratifiedKFold
nfolds = 10

skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
submission = pd.read_csv('sample_submission.csv')

x_train = train_data.tocsr()   # csr is ok to index 
x_test = test_data

y_pred = np.zeros((test.shape[0], 6))
y_preds = {}

xgb_oofs = np.zeros((train.shape[0],6))
for index,cls in enumerate(train.columns[2:]):
    print(cls)
    for i, (tra, val) in enumerate(skf.split(x_train, train[cls])):
        print ("Running Fold", i+1, "/", nfolds, "at class ", cls)
        model = train_and_validate_model(x_train[tra], train[cls][tra], x_train[val], train[cls][val], cls)
        y_pred[:,index] += model.predict_proba(x_test)[:, 1]
        xgb_oofs[val,index] = model.predict_proba(x_train[val])[:, 1]
        
y_pred /= float(nfolds)
xgb_oofs = pd.DataFrame(xgb_oofs)
xgb_oofs.to_csv("xgb_oofs.csv")

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('xgb_tfidf_10foldave.csv', index=False)


# In[23]:


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
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
nfolds = 10

all_parameters = {
                  'C'             : [1.048113, 0.1930, 0.596362, 0.25595, 0.449843, 0.25595],
                  'tol'           : [0.1, 0.1, 0.046416, 0.0215443, 0.1, 0.01],
                  'solver'        : ['lbfgs', 'newton-cg', 'lbfgs', 'newton-cg', 'newton-cg', 'lbfgs'],
                  'fit_intercept' : [True, True, True, True, True, True],
                  'penalty'       : ['l2', 'l2', 'l2', 'l2', 'l2', 'l2'],
                  'class_weight'  : [None, 'balanced', 'balanced', 'balanced', 'balanced', 'balanced'],
                 }

skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
submission = pd.read_csv('sample_submission.csv')

x_train = train_data.tocsr()   # csr is ok to index 
x_test = test_data

y_pred = np.zeros((test.shape[0], 6))
y_preds = {}

from sklearn.metrics import log_loss,roc_auc_score


def train_and_validate_model(model, x_train, y_train, x_validation, y_validation, cls):
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_validation)[:, 1]
    y_pred_tra = model.predict_proba(x_train)[:, 1]
    roc_score = roc_auc_score(y_validation, y_pred)
    val_loss_score = log_loss(y_validation, y_pred)
    train_loss_score = log_loss(y_train, y_pred_tra)
    print("logistic regression: train_loss: {} . val_loss: {}. roc_auc: {}".format(train_loss_score, val_loss_score, roc_score))
    return model


xgb_oofs = np.zeros((train.shape[0],6))
for j,cls in enumerate(train.columns[2:]):
    print(cls)
    for i, (tra, val) in enumerate(skf.split(x_train, train[cls])):
        print ("Running Fold", i+1, "/", nfolds, "at class ", cls)
        if i==0:
            clf = xgb.XGBClassifier(**param)
            clf.fit(x_train[tra], train[cls][tra], 
                    eval_set=[(x_train[tra], train[cls][tra]),(x_train[val], train[cls][val])], eval_metric="auc",early_stopping_rounds=10, verbose=True)
            
            selection = SelectFromModel(clf, threshold="mean", prefit=True)
            select_x_train = selection.transform(x_train)
            
        lr = classifier = LogisticRegression(
                                C=all_parameters['C'][j],
                                max_iter=200,
                                tol=all_parameters['tol'][j],
                                solver=all_parameters['solver'][j],
                                fit_intercept=all_parameters['fit_intercept'][j],
                                penalty=all_parameters['penalty'][j],
                                dual=False,
                                class_weight=all_parameters['class_weight'][j],
                                verbose=0)
        model = train_and_validate_model(lr, select_x_train[tra], train[cls][tra], select_x_train[val], train[cls][val], cls)
        select_x_test = selection.transform(x_test)
        y_pred[:,index] += model.predict_proba(select_x_test)[:, 1]
        xgb_oofs[val,index] = model.predict_proba(select_x_train[val])[:, 1]
        
y_pred /= float(nfolds)
xgb_oofs = pd.DataFrame(xgb_oofs)
xgb_oofs.to_csv("xgb_lr_oofs.csv")

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('xgb_lr_10foldave.csv', index=False)


# In[19]:


y_pred


# In[22]:


submission

