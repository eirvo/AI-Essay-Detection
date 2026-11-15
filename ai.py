from asyncio.windows_utils import pipe
from copyreg import pickle
import pickle

import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

data = 'AI_Human.csv'

random.seed(1)
df = pd.read_csv(data, skiprows=lambda i: i > 0 and random.random() > 0.02)
print(df.shape)
print(df.isna().sum())
print('duplicate essays:', df['text'].duplicated().sum())
print(df['generated'].value_counts(normalize=True))
df['generated'] = df['generated'].astype(int)

df['n_words'] = df['text'].str.split().str.len()
print(df.groupby('generated')['n_words'].describe())

df_full_train, df_test = train_test_split(df, test_size=0.2, stratify=df['generated'], random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.2, stratify=df_full_train['generated'], random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3)
X_train = vec.fit_transform(df_train['text']) 
X_test = vec.transform(df_test['text'])   
X_val = vec.transform(df_val['text'])      

y_train = df_train['generated'].values
y_test = df_test['generated'].values
y_val = df_val['generated'].values

X_full_train = vec.transform(df_full_train['text'])
y_full_train = df_full_train['generated'].values

for C in [0.1, 1, 10, 100]: #tuning the regularization parameter C for logistic regression
    lr = LogisticRegression(C=C, max_iter=1000).fit(X_train, y_train)
    print(C, roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1]))

final = LogisticRegression(C=10, max_iter=1000).fit(X_full_train, y_full_train)

pred = final.predict_proba(X_test)[:, 1]
print('Test AUC:', roc_auc_score(y_test, pred))

words = np.array(vec.get_feature_names_out())
order = np.argsort(final.coef_[0])
print('Most human:', words[order[:20]])
print('Most AI:', words[order[-20:]])

features = list(vec.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


watchlist = [(dtrain, 'train'), (dval, 'val')]
xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc',
              'tree_method': 'hist', 'nthread': 8, 'seed': 1}
best = {'eta': 0.3, 'max_depth': 6, 'min_child_weight': 1}

for name, values in [('eta', [0.1, 0.3]),
                     ('max_depth', [3, 6]),
                     ('min_child_weight', [1, 10])]:
    scores = {}
    for v in values:
        m = xgb.train({**xgb_params, **best, name: v}, dtrain, num_boost_round=150,
                      evals=[(dval, 'val')], early_stopping_rounds=10,
                      verbose_eval=False)
        scores[v] = m.best_score
        print(name, v, 'val AUC:', round(m.best_score, 4))
    best[name] = max(scores, key=scores.get)

print('best params:', best)
params = {**xgb_params, **best}

model = xgb.train(
    params, dtrain,
    num_boost_round=300,
    evals=[(dval, 'validation')],
    early_stopping_rounds=20,
    verbose_eval=25,
)

y_pred = model.predict(dval)
print('Validation AUC:', roc_auc_score(y_val, y_pred))

y_full_train = df_full_train['generated'].astype(int).values
dfull_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

model = xgb.train(
    params, dfull_train,
    num_boost_round=300,
    evals=[(dtest, 'test')],
    early_stopping_rounds=20,
    verbose_eval=25,          
)

y_pred = model.predict(dtest)
print('Test AUC:', roc_auc_score(y_test, y_pred))

with open('model.bin', 'wb') as f:
    pickle.dump(pipe, f)