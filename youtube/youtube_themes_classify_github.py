import os
import json
import warnings

import pandas as pd

# change to the directory with these files
os.chdir('.')

from youtube_themes_github import youtube_clf
from youtube_themes_github import subpreds_to_preds

from random import seed as stdlib_seed
from numpy.random import seed as np_seed


# dont want to see future warnings from pandas/sklearn
warnings.filterwarnings('ignore')


### set rng seeds for reproducibility ----------------------------------------

# (need to set both std lib and np seeds!)
# NOTE: using multithreading invalidates seed state -- watch out!
seed = 6933
np_seed(seed)
stdlib_seed(seed)



### input files --------------------------------------------------------------
hypers_file = 'youtube_themes_hypers.json'

infiles = dict(
  motherloadf = os.path.join('data/prepped/youtube_videos-toy-clean-prepped.csv'))

# NOTE: for * in ['addiction','health','policy','youth'], output files are:
#   - 'data/output/eval_*'
#   - 'data/output/pred_*'



### utilities and globals -----------------------------------------------------
text_col = 'text_blob_scrubbed'
themes = lambda s='': [th+s for th in ['addiction','health','policy','youth']]



### keyword sets for each theme -----------------------------------------------
keywords = dict(
  addiction = ['addicted', 'addiction', 'hooked', 'addict', 'withdrawal'], 
  health = ['health', 'healthy', 'disease', 'hospital',
             'doctor','injured','unhealthy','cancer'], 
  policy = ['illegal', 'legal', 'legislation', 'law', 'congress',
             'government', 'legalize', 'outlaw'], 
  youth = ['kid', 'kids', 'youth', 'highschool', 'underage'])



### hyper parameters for each theme and submodel ------------------------------
with open(hypers_file, 'r') as f:
  hypers = json.load(f)



### read datasets -------------------------------------------------------------
print(f'\nnow reading data files...')

# NOTE: must read with gross char encoding if on norc machine ('ISO-8859-1')
encoding = 'ISO-8859-1'
predset = pd.read_csv(infiles['motherloadf'], encoding=encoding)



### addiction theme -----------------------------------------------------------
print(f'\nnow assigning ensemble model for addiction theme...')

## train submodels and generate subpreds for addiction theme 
addiction_subpreds = youtube_clf(predset, text_col, 
  theme = 'addiction', 
  theme_kw = keywords['addiction'],
  count_vect_hypers = hypers['addiction']['vect']['count'], 
  tfidf_vect_hypers = hypers['addiction']['vect']['tfidf'], 
  sgd_clf_hypers = hypers['addiction']['clf']['sgd'], 
  dtree_clf_hypers = hypers['addiction']['clf']['dtree'])


## aggregate subpreds to generate final predictions 
addiction_predset_preds = subpreds_to_preds(
  addiction_subpreds['predset_subpreds'], theme='addiction', threshold=3)


## write preds to disk 
print(f'writing addiction preds to disk...')
addiction_predset_preds.to_csv('data/output/pred_addiction-.csv', index=False)
print('done with addiction theme.')



### health theme --------------------------------------------------------------
print(f'\nnow assigning ensemble model for health theme...')

## train submodels and generate subpreds for health theme 
health_subpreds = youtube_clf(
  predset, text_col, 
  theme = 'health', 
  theme_kw = keywords['health'],
  count_vect_hypers = hypers['health']['vect']['count'], 
  tfidf_vect_hypers = hypers['health']['vect']['tfidf'], 
  sgd_clf_hypers = hypers['health']['clf']['sgd'], 
  dtree_clf_hypers = hypers['health']['clf']['dtree'])


## aggregate subpreds to generate final predictions 
health_predset_preds = subpreds_to_preds(
  health_subpreds['predset_subpreds'], theme='health', threshold=3)


## write preds to disk 
print(f'writing health preds to disk...')
health_predset_preds.to_csv('data/output/pred_health-.csv', index=False)



### policy theme --------------------------------------------------------------
print(f'\nnow assigning ensemble model for policy theme...')

## train submodels and generate subpreds for policy theme 
policy_subpreds = youtube_clf(
  predset, text_col, 
  theme = 'policy', 
  theme_kw = keywords['policy'],
  count_vect_hypers = hypers['policy']['vect']['count'], 
  tfidf_vect_hypers = hypers['policy']['vect']['tfidf'], 
  sgd_clf_hypers = hypers['policy']['clf']['sgd'], 
  dtree_clf_hypers = hypers['policy']['clf']['dtree'])


## aggregate subpreds to generate final predictions 
policy_predset_preds = subpreds_to_preds(
  policy_subpreds['predset_subpreds'], theme='policy', threshold=3)


## write preds to disk 
print(f'writing policy preds to disk...')
policy_predset_preds.to_csv('data/output/pred_policy-.csv', index=False)



### youth theme ---------------------------------------------------------------
print(f'\nnow assigning ensemble model for youth theme...')

## train submodels and generate subpreds for youth theme 
youth_subpreds = youtube_clf(
  predset, text_col, 
  theme = 'youth', 
  theme_kw = keywords['youth'],
  count_vect_hypers = hypers['youth']['vect']['count'], 
  tfidf_vect_hypers = hypers['youth']['vect']['tfidf'], 
  sgd_clf_hypers = hypers['youth']['clf']['sgd'], 
  dtree_clf_hypers = hypers['youth']['clf']['dtree'])


## aggregate subpreds to generate final predictions 
youth_predset_preds = subpreds_to_preds(
  youth_subpreds['predset_subpreds'], theme='youth', threshold=3)


## write preds to disk 
print(f'writing youth preds to disk...')
youth_predset_preds.to_csv('data/output/pred_youth-.csv', index=False)



print(f'\ndone. video ids with model predictions are now in "data/output/"\n')
