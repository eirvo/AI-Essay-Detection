import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

df = pd.read_json('essays_sample.jsonl', lines=True)
df['generated'] = df['generated'].astype(int)

df_full_train, df_test = train_test_split(
    df, test_size=0.2, stratify=df['generated'], random_state=1)

pipe = make_pipeline(
    TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3),
    LogisticRegression(C=10, max_iter=1000))
pipe.fit(df_full_train['text'], df_full_train['generated'])

print('Test AUC:', roc_auc_score(df_test['generated'],
                                 pipe.predict_proba(df_test['text'])[:, 1]))

with open('model.bin', 'wb') as f:
    pickle.dump(pipe, f)