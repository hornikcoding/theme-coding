import re

import numpy as np
import pandas as pd
import pickle

from functools import partial

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

# take a string `text` and a list of strings `kw_list`, return a boolean
def kw_list_clf(text, kw_list):
  text = re.sub(' +', ' ', re.sub('[^a-zA-Z0-9_ ]+', '', text.strip()))
  words = text.lower().split(' ')
  for kw in kw_list:
    if kw in words:
      return True
  return False



# `trainset`, `evalset`, and `predset` should be data frames 
# `trainset` and `evalset` must have columns:
#   - `text_col`
#   - theme+'_label' for theme in ['addiction','health','policy','youth']
#   - 'video_id'
#  `predset` must have `text_col` and `video_id` 
# 
# see script `youtube_themes_classify.py` for usage examples
def youtube_clf(predset, text_col, theme, theme_kw, 
                count_vect_hypers, tfidf_vect_hypers, 
                sgd_clf_hypers, dtree_clf_hypers):
  
  ### prep data ---------------------------------------------------------------
  print(f'prepping data for theme `{theme}`')
  
  pred_text = predset[text_col]  
  pred_out = predset[['video_id']]  
  
  ### load submodels ---------------------------------------------------------
  
  #load in the features and classifier for each of the 4 classifiers
  ## model 1: restricted count vectors, sgd classifier 
  m1_vect = pickle.load(open('m1_vect_' + theme + '.pkl','rb'))
  m1_clf = pickle.load(open('m1_clf_' + theme + '.pkl','rb'))
  ## model 2: restricted tfidf vectors, sgd classifier 
  m2_vect = pickle.load(open('m2_vect_' + theme + '.pkl','rb'))
  m2_clf = pickle.load(open('m2_clf_' + theme + '.pkl','rb'))
  ## model 3: restricted count vectors, decision tree 
  m3_vect = pickle.load(open('m3_vect_' + theme + '.pkl','rb'))
  m3_clf = pickle.load(open('m3_clf_' + theme + '.pkl','rb'))
  ## model 4: restricted tfidf vectors, decision tree 
  m4_vect = pickle.load(open('m4_vect_' + theme + '.pkl','rb'))
  m4_clf = pickle.load(open('m4_clf_' + theme + '.pkl','rb'))

  
  ## model 5: keyword set for each theme 
  m5_clf = partial(kw_list_clf, kw_list=theme_kw)
    
  
  ### prep unlabeled text -----------------------------------------------------
  print(f'constructing text features for theme `{theme}` unlabeled data')
  
  m1_pred_dtm = m1_vect.transform(pred_text)
  m2_pred_dtm = m2_vect.transform(pred_text)
  m3_pred_dtm = m3_vect.transform(pred_text)
  m4_pred_dtm = m4_vect.transform(pred_text)
  
  
  ### assign unlabeled text predictions ---------------------------------------
  print(f'generating submodel predictions for theme `{theme}` unlabeled data')
  
  pred_out[theme+'_m1_pred'] = m1_clf.predict(m1_pred_dtm)
  pred_out[theme+'_m2_pred'] = m2_clf.predict(m2_pred_dtm)
  pred_out[theme+'_m3_pred'] = m3_clf.predict(m3_pred_dtm)
  pred_out[theme+'_m4_pred'] = m4_clf.predict(m4_pred_dtm)
  pred_out[theme+'_m5_pred'] = [m5_clf(txt) for txt in pred_text]
  
  
  return dict(predset_subpreds=pred_out)


# take a list of booleans `bools`, return True if they sum to >= `threshold`
def aggregate_subpreds(bools, threshold):
  assert all(type(b) in [bool, np.bool_] for b in bools)
  return sum(bools) >= threshold


def subpreds_to_preds(subpred_df, theme, threshold):
  '''take a df of video ids and subpreds and return a df of ids and preds
  params:
    subpred_df: pd.DataFrame, with col `video_id` and the rest boolean subpreds
    theme: a string identifying the theme (will appear in other colnames)
    threshold: how many subpreds need to be true for pred to be true
  usage: 
    subpred_df = pd.read_csv('data/output/pred_health.csv').tail(10)
    subpreds_to_preds(subpred_df, 'health', threshold=2)
  '''
  
  assert 'video_id' in list(subpred_df.columns)  # theme+'_label'
  # assert subpred_df.shape[1] in [6, 7]         # 6 if no labels, 7 if labels
  
  video_ids = list(subpred_df['video_id'])
  
  drop = [s for s in ['video_id', theme+'_label'] if s in subpred_df.columns]
  subpred_m = subpred_df.drop(columns=drop).as_matrix()
  
  preds = []
  for idx in range(len(subpred_m)):
    subpreds = list(subpred_m[idx])
    preds.append(aggregate_subpreds(subpreds, threshold=threshold))
  
  return pd.DataFrame({'video_id': video_ids, 'pred': preds}, 
                      columns=['video_id', 'pred'])



