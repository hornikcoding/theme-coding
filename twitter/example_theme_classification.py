import os
import pandas as pd
import numpy as np

from theme_clf import *



### session setup -------------------------------------------------------------

# REPLACE WITH PATH TO WHEREVER YOU SAVE THIS FOLDER
# make sure your working directory is set to location of `theme_clf.py`!
os.chdir(os.path.join(
  'C:/Users/leffel-timothy/Desktop/projjies/HMC_local/UPenn/twitter/',
  'twitter_regex_theme_classifiers'))


# input/output files 
sample_data_file = 'data/example_data.csv'
outfile = 'data/example_data-with_regex_preds.csv'

# (relative) path to folder holding term lists for each theme 
term_lists_location = 'term_lists'





### example usage -------------------------------------------------------------

### create a classifier by instantiating `ThemeClf()` with a theme string. 
### then use the `.classify_tweet()` method to assign a T/F label to some text. 
### (in this example, you could use `ArticleContent` or `ArticleContent_clean`)
# themes = ['addiction', 'health', 'policy', 'youth']


# read in data 
dat = pd.read_csv(sample_data_file)



### instantiate classifier and generate predictions for 'addiction' theme 
addict_clf = ThemeClf('addiction', term_loc=term_lists_location)
dat['addiction_pred'] = list(map(addict_clf.classify_tweet, dat.ArticleContent))


### instantiate classifier and generate predictions for 'health' theme 
health_clf = ThemeClf('health', term_loc=term_lists_location)
dat['health_pred'] = list(map(health_clf.classify_tweet, dat.ArticleContent))


### instantiate classifier and generate predictions for 'policy' theme 
policy_clf = ThemeClf('policy', term_loc=term_lists_location)
dat['policy_pred'] = list(map(policy_clf.classify_tweet, dat.ArticleContent))


### instantiate classifier and generate predictions for 'youth' theme 
youth_clf = ThemeClf('youth', term_loc=term_lists_location)
# presence of god term results in `True` label no matter what else happens
youth_clf.god_terms = ['high schools?']
dat['youth_pred'] = list(map(youth_clf.classify_tweet, dat.ArticleContent))





### write predictions to file -------------------------------------------------
print('writing data with predictions to: `{outfile}`'.format(outfile=outfile))
dat.to_csv(outfile, index=False)



