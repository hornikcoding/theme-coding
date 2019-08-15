# -*- coding: utf-8 -*-
# theme classifier coding - api-loop [final]
# Laura Gibson
# April 2017
# Revised April 2018 to use text files rather than api loop
#
# use the TCORS database api to pull articles ALREADY CODED AS TOBACCO-RELEVANT, code them and return codes
#See information on the api parameters in \2_Content Analysis\Database-ASC IT\Introduction to the Database.doc


# --- import required modules 

#need to import a newer version in order to do precision & recall division at the end (it thinks handcoded is text)
from __future__ import division

import os
import pandas as pd
import sys
import pickle
import datetime

sys.path.append('C:/Users/lgibson/Desktop/long-form_theme_classifiers')

from utils import class_functions2 as cf

#Change to the appropriate directory 
input_dir = 'C:/Users/lgibson/Desktop/long-form_theme_classifiers'
os.chdir(input_dir) 

# I've moved everything to be in the above directory, so we don't need all of these other locations
#this is where the text to be coded lives
#mdata_dir = '//FILE3/TobaccoSurveyData/P50 - Youth Tobacco Tracking/2_Content Analysis/Data/_Python/Ecig-themes/data/mturk_coded/split'
mdata_dir = input_dir
#this is where you want the output to go
#out_dir_base = '//FILE3/TobaccoSurveyData/P50 - Youth Tobacco Tracking/2_Content Analysis/Data/_Python/Ecig-themes/scripts/Classifiercode_themes/'
out_dir_base = input_dir

version = '_hopkins2_ctob'

#these need to be updated
model = 'log_reg'
vect_type = 'count'
title = True
window=True
param_filename = 'theme_params_norc.csv'
test_filename = 'NORC-data-test.csv'
theme_params = pd.read_csv(os.path.join(input_dir, param_filename))
#theme_params = theme_params[theme_params.theme=='policy']
#clf = 'c' + theme_params.theme.iloc[0] + '1'

if __name__ == "__main__":

    #the cleaning process is doubled up a bit here because I'm mixing API and csv methods - I don't think that causes a problem.
    #get the dataframe from the file and check for duplicates/highlights
    docs = cf.file_chk(os.path.join(mdata_dir, test_filename), title)

    #remove extra spaces and append title to beginning
    docs = cf.norm(docs, title)
    #preprocess abbreviations and make a window
    docs = cf.clean_txt(docs, window)
    
    for index, row in theme_params.iterrows():

        if row.theme == 'addiction':
            row.theme = 'addict'
            
        clf = 'c' + row.theme + '8'
        pickle_dir = clf + version 
        
        if window:
            pickle_spec = pickle_dir + '_win'
        else:
            pickle_spec = pickle_dir

        vect_file = pickle_spec + '_vect.pkl'
        clf_file = pickle_spec + '_class.pkl' 
            
        #load in the features from the final classifier
        vect_final = pickle.load(open(os.path.join(input_dir, vect_file),'rb'))
        #load in the classifier
        clf_final = pickle.load(open(os.path.join(input_dir, clf_file),'rb'))

        #get the scores for the test set 
        test_X_final = vect_final.fit_transform(docs.new_text)
        #get the prediction from the classifier
        class_coded, probas = cf.test(test_X_final, clf_final, model)

        #make the code Boolean for the database
        kcode = clf + '_kcoded'
        probas_yes = clf + '_probas_yes'
        docs[kcode]=class_coded.astype('bool')
        #leave this a decimal for .csv
        docs[probas_yes]=probas
                        
        # name each outfile
        outfilenm = pickle_spec + '_probas.csv'
        #append the article ID and the classify return values to the .csv
        outdocs=docs[['ArticleID', kcode, probas_yes]]             
        outdocs.to_csv(os.path.join(input_dir, outfilenm), index=False)
    
#    currentDT = datetime.datetime.now()
#    print (str(currentDT))     