# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:06:22 2017
Functions to do theme classification

All these...and more!
1. Tools for creating Training and Test datasets
    a. Remove html highlight coding and extra spaces/carriage returns
    b. Divide them 80% for training/20% for test - in some instances (themes), require all of the first round of coding to be in the training set
2. Do pre-processing (single term for FDA/Food and Drug Administration, e-cig/e cigarette, etc.)
3. Make a window instead of using the full text
4. Only use the not passing mentions
5. Set-up text vectorizer
6. Do recursive feature selection with the classifier to find the optimal # features
7. Pickle final vectorizer, classifier, print out features with coefficients to .csv file
8. Get cross-validation metrics for training the classifier, save probabilities to .csv file
    a. N Yes, N No, N Total, Precision, Recall, Optimal # features
9. Get test metrics for test dataset, save probabilities to .csv file
    a. N Yes, N No, N Total, Precision, Recall, r with ave conf, pb-r with ave conf
10. Get estimates for test metrics
    a. Weighted Precision, Weighted Recall, Weighted r with ave conf, Weighted pb-r with ave conf

@author: lgibson 2/6/17
"""

from __future__ import print_function

#cross_validation is deprecated in favor of model_selection - need to update.
#http://scikit-learn.org/stable/modules/cross_validation.html

from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, RFECV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
#from statsmodels.stats.weightstats import DescrStatsW

from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import scipy.sparse as ssp
import re
import os
import sys
#import csv
#import unicodecsv as csv 
#import scikit_learn as sklearn
import pickle
import statsmodels.api as sm

from itertools import compress

#functions
def rm_hlights(texts):
    '''
    Remove html highlight coding and extra spaces/carriage returns
    texts = slice of df containing texts
    '''
    print('\n#### CLEANING TEXT by removing highlights and extra blank characters')
    
    #replace with nothing ''
    html_start = '<span style="background-color: #FFFF00"><b>'
    html_end = '</b></span>'
    nothing = [html_start, html_end]
    #replace with a single space ' '
    returns = '[\n\r\t]'
    spaces = '\s{2,}'    
    space = [returns, spaces]
    
    #replacing with nothing ''
    for s in nothing:
        texts = texts.str.replace(s,'')
    
    #replacing with a single space ' '
    for s in space:
        texts = texts.str.replace(s,' ')

    return texts

def rm_dupes(df, s, fname):
    '''
    Remove duplicates, save them in a separate file html highlight coding and extra spaces/carriage returns
    df = the dataframe to be de-duped
    s = column title on which to de-dupe
    fname = original filename - will be transformed here
    '''
    print('\n#### REMOVING ANY DUPLICATES based on the cleaned text')
 
    #drop duplicate texts (gets most of them)
    docs_dedup = df.drop_duplicates(subset=s)

    #save the duplicates for later checking if there are any
    if docs_dedup.shape[0] < df.shape[0]:
        dupes = pd.concat(g for _, g in df.groupby(s) if len(g) > 1)
        print(dupes.ArticleTitle)
        dupes_filename=fname.replace('.csv','_dupes.csv')
        dupes.to_csv(dupes_filename, encoding='utf-8')

    return docs_dedup

def file_split(df, rand, fname, s, prop=.2):
    '''
    Split the file into training and test sets
    df = the dataframe to be split
    rand = 1 if all texts are randomly selected from database or 0 if not, 
    fname = original filename - will be transformed here
    s = column title that signifies which texts were not randomly selected (here when null for coding)
    prop = test proportion
    '''
    print('\n#### SPLITTING FILE into Train and Test')
    split_dir = '//FILE3/TobaccoSurveyData/P50 - Youth Tobacco Tracking/2_Content Analysis/Data/_Python/Ecig-themes/data/mturk_coded/split'

    if rand==1:
        train, test = train_test_split(df, test_size=prop)
    elif rand==0:
        #separate out the two sets of coded data - where coding is null must go in training set
        df_old = df[df[s].isnull()]
        df_new = df[df[s].notnull()]
        #get N for test prop of total 
        test_ct = int(len(df)*prop)
        train_temp, test = train_test_split(df_new, test_size=test_ct)
        train = pd.concat([train_temp, df_old])

    #print out train/test files for documentation
    train_filename=fname.replace('.csv','_train.csv')
    test_filename=fname.replace('.csv','_test.csv')
    train.to_csv(os.path.join(split_dir, train_filename), encoding='utf-8')
    test.to_csv(os.path.join(split_dir, test_filename), encoding='utf-8')

    return train, test

def file_split_rev(df, fname, prop=.2):
    '''
    Split the file into training and test sets
    df = the dataframe to be split
    fname = original filename - will be transformed here
    prop = test proportion
    '''
    print('\n#### SPLITTING FILE into Train and Test')
    split_dir = '//FILE3/TobaccoSurveyData/P50 - Youth Tobacco Tracking/2_Content Analysis/Data/_Python/Ecig-themes/data/mturk_coded/split'

    train, test = train_test_split(df, test_size=prop)

    #print out train/test files for documentation
    train_filename=fname.replace('.csv','_train.csv')
    test_filename=fname.replace('.csv','_test.csv')
    train.to_csv(os.path.join(split_dir, train_filename), encoding='utf-8')
    test.to_csv(os.path.join(split_dir, test_filename), encoding='utf-8')

    return train, test


def preprocess(texts):
    '''
    Replace terms like FDA and Food and Drug Administration with 1 term: Food_and_drug_administration; also done for e-cig variants
    texts = slice of df containing cleaned texts
    '''
    print('\n#### DOING PREPROCESSING - replacing terms with abbreviations')
    # 1. load abbreviations mapping file
    map_file = pd.read_csv('abbreviations_final.csv')

    # 2. create mapping dictionary
    re_dict={}
    for gr in map_file.groupby('Common Label', as_index=False):
        matches = list(gr[1]['Feature Variants'])
        matches.sort(key=lambda x: len(x), reverse=True)
        rexp = '|'.join(['\\b%s\\b' % m for m in matches])
        re_dict[gr[0]] = r'%s' % rexp

    for rep, match in re_dict.items():
        print(match, rep)
        texts = texts.str.replace(match,rep,flags=re.I) 
    return texts

def poslist(warray, feats, window=20):
    '''
    find each start point for matching features (highlights, not just tm)
    get start and end points 20 +/-
        if that distance is >= 20 words, return 1
    warray = a text split into an ordered list of words
    feats = words to be matched
    window = 20 words before and after the matching word. Can be adjusted
    '''
   
    word_pos = [i for i, x in enumerate(warray) if re.match(feats,x,re.IGNORECASE)]

    groups = []

    while len(word_pos)>0:
        # get first item
        current_item = word_pos.pop(0)
        
        current_start_point = current_item - window if current_item>=window else 0

        while len(word_pos)>0 and word_pos[0] <= current_item+window*2:
            current_item = word_pos.pop(0)
        
        current_end_point = current_item+window
                     
        current_group = [current_start_point, current_end_point]
        
        groups.append(current_group)
        
    return groups                   

def windows(texts):
    '''
    reduce text to just that around the tobacco maybe words
    texts = slice of df containing texts
    '''
    print('\n#### FINDING WINDOWS AROUND KEYWORDS - shortening the text')

    tm=r'(\b(smoking|tobacco|smokers?|smokes?|nicotine|hookahs?|cigs?|e-?cigs?|vaping|cigars?|cigarettes?|cigarillos?|e-?cigars?|e-?cigarillos?|e-?cigarettes?|e_cigarette|chewing|dipping|snus|snuff|smokeless)\b|\b(vape))'
    win_texts = []
    countall = []
    for row in texts:
        wsplit = row.split()
        groups = poslist(wsplit,tm)
        countall.append(len(groups))
        chunks = [' '.join(wsplit[group[0]:group[1]+1]) for group in groups]    
        win_texts.append(' '.join(chunks))

    return win_texts, countall

def chunk_split(docs):

    new_data=[]
    texts = docs.to_dict('records')
    tm=r'(\b(smoking|tobacco|smokers?|smokes?|nicotine|hookahs?|cigs?|e-?cigs?|vaping|cigars?|cigarettes?|cigarillos?|e-?cigars?|e-?cigarillos?|e-?cigarettes?|e_cigarette|chewing|dipping|snus|snuff|smokeless)\b|\b(vape))'
    for row in texts:
        wsplit = row['new_text'].split()
        groups = poslist(wsplit,tm)
        chunks = [' '.join(wsplit[group[0]:group[1]+1]) for group in groups]    
        for cidx, chunk in enumerate(chunks):
            new_row = row.copy()
            new_row['chunk']=chunk
            new_row['chunkID']=cidx+1
            new_data.append(new_row)

    new_docs = pd.DataFrame.from_dict(new_data)

    return new_docs
    
def limit(df, outcome):
    '''
    there are missing values for THEME2cat_cut80 (the middle that gets dropped)
    limiting the dataframe to exclude missing values
    df = dataframe to be limited
    outcome = hand-coding criterion
    '''
    print('\n#### LIMITING TEXT - based on outcome')

    df = df[df[outcome].notnull()]
    y = df[outcome]

    print(df.shape)
    print(pd.crosstab(df.SourceTitle, df[outcome]))

    return df, y

def norm(docs, title):
    '''
    check that api files do not have highlights, extra spaces, 
    '''
    print('\n#### CHECKING FILE - remove highlights and extra spaces')
    print(docs.shape)

    #remove highlights and extra spaces
    docs['new_text'] = rm_hlights(docs.ArticleContent)
    #print one example to see what changed
    print(docs.ArticleContent.head(5))
    print('\n',docs.new_text.head(5))

    if title:
        s = rm_hlights(docs.ArticleTitle)
        s = s.replace(np.nan, ' ', regex=True)
        docs.new_text = s + ' ' + docs.new_text
        print('\n',docs.new_text.head(5))

    return docs

def clean_txt(docs, window=False):
    '''
    pre-process data and limit to the dataset you want
    '''
    texts=docs.new_text
    #pre-process to clean up abbreviations
    texts = preprocess(texts)

    #reduce text to windows if you want
    if window:
        texts, countall = windows(texts)
        print(texts[0])
        docs['win_ct'] = countall
        print(docs.win_ct.sum())
        print(docs.win_ct[docs.win_ct>1].sum())

    docs['new_text'] = texts

    return docs

def file_chk(filename, title):
    '''
    check that files do not have highlights, extra spaces, duplicates, or missing new_text
    '''
    print('\n#### CHECKING FILE - remove highlights and dupes')
    docs = pd.read_csv(filename)
    print(docs.shape)

    #remove highlights and extra spaces
    docs['new_text'] = rm_hlights(docs.ArticleContent)
    #print one example to see what changed
    print(docs.ArticleContent.head(5))
    print('\n',docs.new_text.head(5))

    #remove duplicates if there are any and return de-duplicated texts
    docs = rm_dupes(docs, 'new_text', filename)
    print(docs.shape)

    if title:
        s = rm_hlights(docs.ArticleTitle)
        s = s.replace(np.nan, ' ', regex=True)
#        docs.new_text = s + ' ' + docs.new_text
        docs.new_text = s.fillna('') + ' ' + docs.new_text.fillna('')
        print('\n',docs.new_text.head(5))

    return docs

def file_chk_new_text(filename, title):
    '''
    check that files do not have highlights, extra spaces, duplicates, or missing new_text
    for use if there is no ArticleContent
    '''
    print('\n#### CHECKING FILE - remove highlights and dupes')
    docs = pd.read_csv(filename)
    print(docs.shape)

    print(docs.new_text.head(5))
    #remove highlights and extra spaces
    docs['new_text'] = rm_hlights(docs.new_text)
    #print one example to see what changed
    print('\n',docs.new_text.head(5))

    #remove duplicates if there are any and return de-duplicated texts
    docs = rm_dupes(docs, 'new_text', filename)
    print(docs.shape)

    if title:
        s = rm_hlights(docs.ArticleTitle)
        s = s.replace(np.nan, ' ', regex=True)
#        docs.new_text = s + ' ' + docs.new_text
        docs.new_text = s.fillna('') + ' ' + docs.new_text.fillna('')
        print('\n',docs.new_text.head(5))

    return docs

def clean_df(docs, scode, window=False, npm=False, pm=False, nobtn=False):
    '''
    pre-process data and limit to the dataset you want
    '''
    texts=docs.new_text
    #pre-process to clean up abbreviations
    texts = preprocess(texts)

    #reduce text to windows if you want
    if window:
        texts, countall = windows(texts)
        print(texts[0])
        docs['win_ct'] = countall
        print(docs.win_ct.sum())
        print(docs.win_ct[docs.win_ct>1].sum())

    docs['new_text'] = texts

    #reduce texts to not passing mentions only if you want
    #MAKE SURE THE OLD DATA FILES HAVE NPM CODED (this makes it look like they won't)
    if npm:
        docs = docs.loc[docs['kw_coded_npm'] == True]
        print(docs.shape)
    elif pm:
        docs = docs.loc[docs['kw_coded_npm'] == False]
        print(docs.shape)

    #same but limit to particular sources (i.e., exclude BTN here)
    if nobtn:
        docs = docs[docs.Source.isin(['ap','news','web'])]
        print(docs.shape)

    #limit to not missing on the outcome
    docs, y = limit(docs, scode)
        
    return docs, y

from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer(object):
    def __init__(self):
         self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
    
def vectorize(texts, vect_type):
    '''
    Set-up text vectorizer
    texts = slice of df containing text to be trained on (cleaned or windowed)
    vect_type = CountVectorizer of TfidfVectorizer
    '''
    print('\n#### VECTORIZE TRAINING DATASET')

    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            stop_words='english',
                            min_df=3,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
                            binary = True,
                            )
    elif vect_type=='tfidf':
        #not using this one right now
        vect1 = TfidfVectorizer(decode_error='ignore',
                             stop_words='english',
                             min_df=3,
                             sublinear_tf=True,
                             ngram_range=(1,3),
                             token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
                             binary = False
                            )
   
    #this is a sparse matrix in coordinate format
    matrix = vect1.fit_transform(texts)
    return matrix, vect1

def vectorize_nostop(texts, vect_type):
    '''
    Set-up text vectorizer
    texts = slice of df containing text to be trained on (cleaned or windowed)
    vect_type = CountVectorizer of TfidfVectorizer
    '''
    print('\n#### VECTORIZE TRAINING DATASET')

    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            min_df=3,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
                            binary = True,
                            )
    elif vect_type=='tfidf':
        #not using this one right now
        vect1 = TfidfVectorizer(decode_error='ignore',
                             min_df=3,
                             sublinear_tf=True,
                             ngram_range=(1,3),
                             token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
                             binary = False
                            )
   
    #this is a sparse matrix in coordinate format
    matrix = vect1.fit_transform(texts)
    return matrix, vect1

def vectorize_hop(texts, vect_type):
    '''
    Set-up text vectorizer
    texts = slice of df containing text to be trained on (cleaned or windowed)
    vect_type = CountVectorizer of TfidfVectorizer
    
#    vectorizer_hop1
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
#                            ngram_range=(1,3),
#                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
                            tokenizer = tokenize,                            
                            binary = True,
                            )

    
#    vectorizer_hop2       
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    
#    vectorizer_hop3
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
#                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
#   vectorizer_hop4
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
#   vectorizer_hop5
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
#   vectorizer_hop6
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = False,
                            )
#    vectorizer_hop7       
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = False,
                            )

#    vectorizer_hop8       
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.98,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    '''

    print('\n#### VECTORIZE TRAINING DATASET')

    stemmer = PorterStemmer()

    # based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed
    
    def tokenize(text):
        tokens = word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems

#    vectorizer hop2
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    elif vect_type=='tfidf':
        #not using this one right now
        vect1 = TfidfVectorizer(decode_error='ignore',
                             stop_words='english',
                             min_df=3,
                             sublinear_tf=True,
                             ngram_range=(1,3),
                             token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
                             binary = False
                            )
   
    #this is a sparse matrix in coordinate format
    matrix = vect1.fit_transform(texts)
    return matrix, vect1

def vectorize_ls(texts, vect_type):
    '''
    Set-up text vectorizer
    texts = slice of df containing text to be trained on (cleaned or windowed)
    vect_type = CountVectorizer of TfidfVectorizer
    
#    vectorizer_hop1
        if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
#                            ngram_range=(1,3),
#                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
                            tokenizer = tokenize,                            
                            binary = True,
                            )

    
#    vectorizer_hop2       
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    
    vectorizer_hop3
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
#                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    vectorizer_hop4
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    '''


    print('\n#### VECTORIZE TRAINING DATASET')

    stemmer = PorterStemmer()

    # based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed
    
    def tokenize(text):
        tokens = word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems

#    vectorizer_hop2       
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    
    elif vect_type=='tfidf':
        #not using this one right now
        vect1 = TfidfVectorizer(decode_error='ignore',
                             stop_words='english',
                             min_df=3,
                             sublinear_tf=True,
                             ngram_range=(1,3),
                             token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
                             binary = False
                            )
   
    #this is a sparse matrix in coordinate format
    matrix = vect1.fit_transform(texts)
    return matrix, vect1

#tobacco/ecig valence vectorizer: 10/01/2017 kkim modified lgibson's syntax
def vectorize_kk(texts, vect_type):
    '''
    Set-up text vectorizer
    texts = slice of df containing text to be trained on (cleaned or windowed)
    vect_type = CountVectorizer of TfidfVectorizer
    
#    vectorizer_kk1
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
#                            ngram_range=(1,3),
#                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
                            tokenizer = tokenize,                            
                            binary = True,
                            )

    
#    vectorizer_kk2       
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    
#    vectorizer_kk3
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
#                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
#   vectorizer_kk4
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
#   vectorizer_kk5
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
#   vectorizer_kk6
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = False,
                            )
#    vectorizer_kk7       
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = False,
                            )

#    vectorizer_kk8       
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
#                            stop_words='english',
                            max_df=.98,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    '''

    print('\n#### VECTORIZE TRAINING DATASET')

    stemmer = PorterStemmer()

    # based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed
    
    def tokenize(text):
        tokens = word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems

#   vectorizer_kk4
    if vect_type=='count':
        # 1. set up text vectorizer
        vect1 = CountVectorizer(decode_error='ignore',
                            stop_words='english',
                            max_df=.99,
                            min_df=.01,
                            ngram_range=(1,3),
                            token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                            tokenizer=LemmaTokenizer(),
#                            tokenizer = tokenize,                            
                            binary = True,
                            )
    elif vect_type=='tfidf':
        #not using this one right now
        vect1 = TfidfVectorizer(decode_error='ignore',
                             stop_words='english',
                             min_df=3,
                             sublinear_tf=True,
                             ngram_range=(1,3),
                             token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
                             binary = False
                            )
   
    #this is a sparse matrix in coordinate format
    matrix = vect1.fit_transform(texts)
    return matrix, vect1

def rfs(matrix, y, vect, clfname, model, max_feats=5000, steps=100, folds=5):
    '''
    Do recursive feature selection with the classifier to find the optimal # features
    Returns Optimal # feats
    
    matrix, y, vect, clfname, model
    
    '''

    print('\n#### RECURSIVE FEATURE SELECTION - to get optimal # features')

    #feature selection
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
    print(matrix.shape[1])
    NUM_OF_FEATURES = min(matrix.shape[1], max_feats)    # minimum of total num of feats from vectorizer and user specified
    selector = SelectKBest(chi2, k=NUM_OF_FEATURES)
    X=selector.fit_transform(matrix, y)

    #create classifier and fit model
    if model=='log_reg':
        #classifier = LogisticRegression()
        #classifier = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
        #best for addiction
        #classifier = LogisticRegression(class_weight={0:.9,1:.1})
        #classifier = LogisticRegression(C=1.0, penalty='l1', tol=0.01, class_weight={0:.9,1:.1})
        #best for health
        #classifier = LogisticRegression(class_weight={0:.8,1:.2})
        #classifier = LogisticRegression(C=1.0, penalty='l1', tol=0.01, class_weight={0:.8,1:.2})
        #currently used in the paper
        classifier = LogisticRegression(class_weight='balanced')
    elif model=='onemany':
        classifier = OneVsRestClassifier(LogisticRegression(class_weight='balanced'))
    elif model=='lin_reg':
    #use LinearSVR for continuous predicting continuous
        classifier = LinearSVR()
    elif model=='svm':
        classifier = LinearSVC()
    elif model=='sgd':
        classifier = SGDClassifier(loss='log')
    
    #provide a random seed so that the same 5-folds are used each time.    
    feature_selection = RFECV(estimator=classifier, step=steps, cv=StratifiedKFold(folds), scoring='f1')
    #this will give you a different set of features every time because the folds are divided differently every time.
    #we are not doing this because it is better to compare across methods (npm, win, etc.) with the same exact folds
    #feature_selection = RFECV(estimator=classifier, step=steps, cv=StratifiedKFold(folds,shuffle=True), scoring='f1')
    
    #knows to fit the best features
    rfecv = feature_selection.fit(X,y)

    #graph the f1 score for each # features
    feat_range = list(range(NUM_OF_FEATURES, 0, -steps))
    feat_range.reverse()
    feat_range.insert(0,0)

    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (f1)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.xticks(range(1, len(rfecv.grid_scores_) + 1), feat_range, rotation=90)
    ax.xaxis.set_ticks_position('top')

    plt.show()

    #print the f1 score for each # features
    print(rfecv.grid_scores_)

    max_score = max(rfecv.grid_scores_)
    best_feat_num = np.where(rfecv.grid_scores_ == max_score)[0][0]

    print('\n\nBEST NUM OF FEATURES WAS {} WITH F1 SCORE OF {}'.format(feat_range[best_feat_num], max_score))

    # best features 
    # old version that stopped working 
#    features = selector.transform(vect.get_feature_names())[0][rfecv.support_]
    # new version as of 1/22/2018
    features = list(compress(vect.get_feature_names(),rfecv.support_))
    print(features, len(features))
    
    return feat_range[best_feat_num], max_score, features, classifier

def kbestfs(matrix, y, vect, clfname, model, max_feats=100):
    '''
    Do k-best feature selection with the classifier to find the features
    Returns feats +
    '''

    print('\n#### K-BEST FEATURE SELECTION - to get features')

    #feature selection
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
    print(matrix.shape[1])
    NUM_OF_FEATURES = min(matrix.shape[1], max_feats)    # minimum of total num of feats from vectorizer and user specified
    selector = SelectKBest(chi2, k=NUM_OF_FEATURES)
    X=selector.fit_transform(matrix, y)

    #create classifier and fit model
    if model=='log_reg':
        #classifier = LogisticRegression()
        #classifier = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
        #classifier = LogisticRegression(class_weight={0:.1,1:.9})
        classifier = LogisticRegression(class_weight='balanced')
    elif model=='lin_reg':
    #use LinearSVR for continuous predicting continuous
        classifier = LinearSVR()
    elif model=='svm':
        classifier = LinearSVC()
    elif model=='sgd':
        classifier = SGDClassifier(loss='log')
    
    classifier.fit(X,y)

    feature_names = vect.get_feature_names()
    features = [feature_names[i] for i in selector.get_support(indices=True)]

    # best features 
    print(features, type(features))

    #potentially use crossvalidate - but we do that later in the script anyway
    #cv, predicted, probas=crossvalidate(X, classifier, y, model)
    '''    
    #can't do this part with 3 category logistic regression classifier
    cv=cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='f1')
    score = np.mean(cv)
    
    print('\n\nF1 SCORE FOR {} FEATURES WAS {}'.format(max_feats, score))
    
    return max_feats, score, features, classifier
    '''
    return max_feats, features, classifier

def vectored(feats, clfname, vect_type):
    '''
    Pickle final text vectorizer using the optimal # features, 
    feats = all features used in this classifier
    clfname = filename for this classifier (e.g., caddict5.csv)
    vect_type = CountVectorizer of TfidfVectorizer
    '''
    print('\n#### VECTORED FEATURES')
    if vect_type=='count':
        vect2 = CountVectorizer(decode_error='ignore',
                        ngram_range=(1,3),
                        token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
#                        tokenizer=LemmaTokenizer(),
                        binary = True,
                        vocabulary = feats,
                        )
    elif vect_type=='tfidf':
        vect2 = TfidfVectorizer(decode_error='ignore',
                        sublinear_tf=True,
                        ngram_range=(1,3),
                        token_pattern='(?u)\\b\\w(?:\\w|[\']\\w)+\\b',
                        binary = False,
                        vocabulary = feats,
                        )

    #save the final count vectorizer to use elsewhere
    vect_filename = clfname.replace('.csv', '_vect.pkl')
    pickle.dump(vect2, open(vect_filename,'wb'))

    return vect2

def classified(clf, X_final, y, clfname):
    '''
    Pickle final classifier using the optimal # features, 
    '''
    print('\n#### FINAL CLASSIFIER')
    #after you do cross-validation, want the weights from applying the classifier across all cases
    clf2=clf.fit(X_final,y)

    #save the final classifier to use elsewhere
    clf_filename = clfname.replace('.csv', '_class.pkl')
    pickle.dump(clf2, open(clf_filename,'wb'))
    
    return clf2

def classified_smp_wt(clf, X_final, y, wt, clfname):
    '''
    Pickle final classifier using the optimal # features, 
    '''
    print('\n#### FINAL CLASSIFIER')
    #after you do cross-validation, want the weights from applying the classifier across all cases
    clf2=clf.fit(X_final,y, sample_weight=wt)

    #save the final classifier to use elsewhere
    clf_filename = clfname.replace('.csv', '_class.pkl')
    pickle.dump(clf2, open(clf_filename,'wb'))
    
    return clf2


def print_feats(vect2, clf2, clfname):
    '''
    print out features with coefficients to .csv file
    '''
    print('\n#### PRINT FEATURES WITH WEIGHTS')
    #can't get the tfidf final vector to work without fitting it again - I don't think that's right.
    #X_final = tfidf_vect_final.fit_transform(docs.ArticleContent_new) 

    feats_coef = pd.DataFrame({'feat':vect2.vocabulary,'coef':clf2.coef_[0]})
    result = feats_coef.sort_values(by='coef',ascending=0)
    feats_filename=clfname.replace('.csv','_featsw.csv')
    result.to_csv(feats_filename, encoding='utf-8')

def crossvalidate(X_final, clf, y, model, folds=5):
    '''
    Get cross-validation metrics for training the classifier, save probabilities to .csv file
    Return: predicted 
    '''
    print('\n#### DOING FINAL CROSS-VALIDATION - with optimal number of features')

    #get the matrix with this list of features
#   MATT - this is giving me a deprecation warning: 
#   Passing 1d arrays as data is deprecated in 0.17 and willraise ValueError in 0.19. Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) if it contains a single sample.
    #f1 scores from each fold
    cv=cross_validation.cross_val_score(clf, X_final, y, cv=folds, scoring='f1')
    print('\n\nF1 for each of', folds,'folds with optimal number of features', cv, 'Ave F1 =', np.mean(cv))

    # get predicted labels (0,1)
    #returns, for each element in the input, the prediction that was obtained for that element when it was in the test set
    predicted=cross_validation.cross_val_predict(clf, X_final, y, cv=folds)

    if model=='log_reg':
        #wrapper in order to get cross validation probabilities - can't do it with class weights...
        class proba_logreg(LogisticRegression):
            def predict(self, X_final):
                return LogisticRegression.predict_proba(self,X_final)

        #cross-validation probabilities
        probas = cross_validation.cross_val_predict(proba_logreg(), X_final, y, cv=folds)
        #print(probas)

        probas = probas[:,1]
    elif model=='onemany':
        ##MATT - I NEED HELP HERE - CAN USE probability=True potentially?
        #http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
        #cross-validation probabilities
        probas = cross_validation.cross_val_predict(proba_logreg(), X_final, y, cv=folds)
        #print(probas)
        probas = probas[:,1]
        
    return cv, predicted, probas

def metrics_bob(docs, y, handcode_full, model, clfname, tt):
    '''
    Get metrics for the classifier, save probabilities to .csv file
    Return: N Yes, N No, N Total, Precision, Recall
    '''
    print('\n#### GETTING METRICS')
    #metrics report 
    metrics_report = metrics.precision_recall_fscore_support(y, docs.pred)
    ''' this gives us
    (array([ 0.89722222,  0.80952381]),  precision for NO YES
           array([ 0.95845697,  0.61658031]),  recall for NO YES
                 array([ 0.92682927,  0.7       ]), F1 for NO YES
                       array([674, 193])) cases support NO YES
    '''
    #note these won't be the same as F1 from step-down process in cross-validation because fold is randomly assigned
    prec = metrics_report[0][1]
    rec = metrics_report[1][1]
    f1 = metrics_report[2][1]
    yes = metrics_report[3][1]
    no = metrics_report[3][0]
    pbr = docs['pred'].corr(docs[handcode_full])

    if model=='log_reg':
        #Printing a graph of the predicted vs. actual
        fig, ax = plt.subplots()
        ax.scatter(y, docs.pred_proba)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
        
        r = docs['pred_proba'].corr(docs[handcode_full])
        
        dfout = docs.copy()
        del dfout['new_text']
        if 'ArticleContent' in dfout.columns:
            del dfout['ArticleContent']

        if tt=='train':
            outfilenm = clfname.replace('.csv', '_pred-train.csv')
        elif tt=='test':
            outfilenm = clfname.replace('.csv', '_pred-test.csv')
        elif tt=='test-all':
            outfilenm = clfname.replace('.csv', '_pred-test-all.csv')
        elif tt=='test_any':
            outfilenm = clfname.replace('.csv', '_pred-test_anychunk.csv')
        elif tt=='test_50':
            outfilenm = clfname.replace('.csv', '_pred-test_50chunkproba.csv')
        dfout.to_csv(outfilenm, encoding='utf-8')

    print('\nPrecision for NO YES =',prec,'\nRecall for NO YES =',rec,'\nF1 for NO YES =',f1,'\nN training cases NO =',no,' YES =', yes,' Total =',no+yes,'\nCorrelation',r,'\nPoint-biserial correlation',pbr)

    return {'yes': yes, 'no':no, 'prec': prec, 'rec': rec, 'f1':f1, 'pbr':pbr, 'r':r, 'tt':tt, 'clfname':clfname}

def test(X_final, clf, model):
    '''
    Get predicted and probabilities for the test dataset
    '''
    print('\n#### CLASSIFYING THE TEST SET')

    #print test
    test_predicted=clf.predict(X_final)
    if model=='log_reg':
        test_probas = clf.predict_proba(X_final)

    test_probas=test_probas[:,1]
    
    return test_predicted, test_probas

def add_weights0(df, weights, theme):
    weights = weights.loc[weights['theme']==theme]
    new_df = pd.merge(df,weights, how='left', on=['SourceTitle'])
    return new_df

def add_weights(df, weights, theme, cueold):
    df = df.rename(columns={cueold: 'coding'})
    weights = weights.loc[weights['theme']==theme]
    new_df = pd.merge(df,weights, how='left', on=['SourceTitle','coding'])
    return new_df

def add_weights2(df, weights, theme, cueold):
    weights = weights.loc[weights['theme']==theme]
    new_df = pd.merge(df,weights, how='left', on=['SourceTitle','coding','level_0'])
    return new_df

def correls_nocat(handcode_full, model, clfname, tt):
    if model=='log_reg':
        if tt=='test':
            outfilenm = clfname.replace('.csv', '_pred-test.csv')
        elif tt=='test-all':
            outfilenm = clfname.replace('.csv', '_pred-test-all.csv')
        df = pd.read_csv(outfilenm)
        r = df['pred_proba'].corr(df[handcode_full])
        total = df.shape[0]
    print('\nTotal =',total,'\nCorrelation', r)
    return {'yes':total, 'no':0,'r':r,'tt':tt, 'clfname':clfname}

def wt_correls_nocat(handcode_full, model, clfname, weights, theme, cueold, tt):
    if model=='log_reg':
        if tt=='test':
            outfilenm = clfname.replace('.csv', '_pred-test.csv')
        elif tt=='test-all':
            outfilenm = clfname.replace('.csv', '_pred-test-all.csv')
        df = pd.read_csv(outfilenm)
        wt_df = add_weights(df,weights,theme,cueold)
        print(wt_df.shape)
        #try making just a df for these two columns
        wt_df2 = wt_df[['pred_proba',handcode_full]].copy()
        r_wt = sm.stats.DescrStatsW(wt_df2, weights=wt_df['weight'])
        total_wt = r_wt.sum_weights
    print('\nWeighted Total =',total_wt,'\nWeighted correlation',r_wt.corrcoef[0,1])

    tt=tt+'_wt'
    return {'yes':total_wt,'no':0,'r':r_wt.corrcoef[0,1],'tt':tt, 'clfname':clfname}

def wt_metrics_bob0(handcode_2bin, handcode_full, model, clfname, weights, theme, tt):

    print('\n#### GETTING WEIGHTED METRICS')
    #metrics report 
#    metrics_report = metrics.precision_recall_fscore_support(y, df.pred)
    #note these won't be the same as F1 from step-down process in cross-validation because fold is randomly assigned
#    prec = metrics_report[0][1]
#    rec = metrics_report[1][1]
#    f1 = metrics_report[2][1]
#    yes = metrics_report[3][1]
#    no = metrics_report[3][0]
#    pbr = df['pred'].corr(df[handcode_full])

    if model=='log_reg':
        if tt=='test':
            outfilenm = clfname.replace('.csv', '_pred-test.csv')
        elif tt=='test-all':
            outfilenm = clfname.replace('.csv', '_pred-test-all.csv')
        
        df = pd.read_csv(outfilenm)
        wt_df = add_weights0(df,weights,theme)
        print(wt_df.shape)
        #try making just a df for these two columns
        wt_df1 = wt_df[['pred',handcode_full]].copy()
        pbr_wt = sm.stats.DescrStatsW(wt_df1, weights=wt_df['weight'])
        wt_df2 = wt_df[['pred_proba',handcode_full]].copy()
        r_wt = sm.stats.DescrStatsW(wt_df2, weights=wt_df['weight'])
        outfilenmw = outfilenm.replace('.csv', 'w.csv')
        wt_df.to_csv(outfilenmw, encoding='utf-8')
        
        wt_metrics_report = metrics.precision_recall_fscore_support(wt_df[handcode_2bin], wt_df.pred, sample_weight=wt_df.weight)

        prec_wt = wt_metrics_report[0][1]
        rec_wt = wt_metrics_report[1][1]
        f1_wt = wt_metrics_report[2][1]
        yes_wt = wt_metrics_report[3][1]
        no_wt = wt_metrics_report[3][0]
 
#    print('\nWeighted correlation',pbr_wt.corrcoef[0,1])
    print('\nWeighted precision =',prec_wt,'\nWeighted recall =',rec_wt,'\nWeighted F1 =',f1_wt,'\nWeighted N training cases NO =',no_wt,' YES =', yes_wt,' Total =',no_wt+yes_wt,'\nWeighted correlation',r_wt.corrcoef[0,1],'\nWeighted point-biserial correlation',pbr_wt.corrcoef[0,1])

    tt=tt+'_wt'
    return {'yes': yes_wt, 'no':no_wt, 'prec': prec_wt, 'rec': rec_wt, 'f1':f1_wt, 'pbr':pbr_wt.corrcoef[0,1], 'r':r_wt.corrcoef[0,1],'tt':tt, 'clfname':clfname}


def wt_metrics_bob(handcode_2bin, handcode_full, model, clfname, weights, theme, cueold, tt):

    print('\n#### GETTING WEIGHTED METRICS')
    #metrics report 
#    metrics_report = metrics.precision_recall_fscore_support(y, df.pred)
    #note these won't be the same as F1 from step-down process in cross-validation because fold is randomly assigned
#    prec = metrics_report[0][1]
#    rec = metrics_report[1][1]
#    f1 = metrics_report[2][1]
#    yes = metrics_report[3][1]
#    no = metrics_report[3][0]
#    pbr = df['pred'].corr(df[handcode_full])

    if model=='log_reg':
        if tt=='test':
            outfilenm = clfname.replace('.csv', '_pred-test.csv')
        elif tt=='test-all':
            outfilenm = clfname.replace('.csv', '_pred-test-all.csv')
        
        df = pd.read_csv(outfilenm)
        wt_df = add_weights(df,weights,theme,cueold)
        print(wt_df.shape)
        #try making just a df for these two columns
        wt_df1 = wt_df[['pred',handcode_full]].copy()
        pbr_wt = sm.stats.DescrStatsW(wt_df1, weights=wt_df['weight'])
        wt_df2 = wt_df[['pred_proba',handcode_full]].copy()
        r_wt = sm.stats.DescrStatsW(wt_df2, weights=wt_df['weight'])
        outfilenmw = outfilenm.replace('.csv', 'w.csv')
        wt_df.to_csv(outfilenmw, encoding='utf-8')
        
        wt_metrics_report = metrics.precision_recall_fscore_support(wt_df[handcode_2bin], wt_df.pred, sample_weight=wt_df.weight)

        prec_wt = wt_metrics_report[0][1]
        rec_wt = wt_metrics_report[1][1]
        f1_wt = wt_metrics_report[2][1]
        yes_wt = wt_metrics_report[3][1]
        no_wt = wt_metrics_report[3][0]
 
#    print('\nWeighted correlation',pbr_wt.corrcoef[0,1])
    print('\nWeighted precision =',prec_wt,'\nWeighted recall =',rec_wt,'\nWeighted F1 =',f1_wt,'\nWeighted N training cases NO =',no_wt,' YES =', yes_wt,' Total =',no_wt+yes_wt,'\nWeighted correlation',r_wt.corrcoef[0,1],'\nWeighted point-biserial correlation',pbr_wt.corrcoef[0,1])

    tt=tt+'_wt'
    return {'yes': yes_wt, 'no':no_wt, 'prec': prec_wt, 'rec': rec_wt, 'f1':f1_wt, 'pbr':pbr_wt.corrcoef[0,1], 'r':r_wt.corrcoef[0,1],'tt':tt, 'clfname':clfname}

def wt_metrics_bob2(handcode_2bin, handcode_full, model, clfname, weights, theme, cueold, tt):

    print('\n#### GETTING WEIGHTED METRICS')
    #metrics report 
#    metrics_report = metrics.precision_recall_fscore_support(y, df.pred)
    #note these won't be the same as F1 from step-down process in cross-validation because fold is randomly assigned
#    prec = metrics_report[0][1]
#    rec = metrics_report[1][1]
#    f1 = metrics_report[2][1]
#    yes = metrics_report[3][1]
#    no = metrics_report[3][0]
#    pbr = df['pred'].corr(df[handcode_full])

    if model=='log_reg':
        if tt=='test':
            outfilenm = clfname.replace('.csv', '_pred-test.csv')
        elif tt=='test-all':
            outfilenm = clfname.replace('.csv', '_pred-test-all.csv')
        
        df = pd.read_csv(outfilenm)
        wt_df = add_weights2(df,weights,theme,cueold)
        print(wt_df.shape)
        #try making just a df for these two columns
        wt_df1 = wt_df[['pred',handcode_full]].copy()
        pbr_wt = sm.stats.DescrStatsW(wt_df1, weights=wt_df['weight'])
        wt_df2 = wt_df[['pred_proba',handcode_full]].copy()
        r_wt = sm.stats.DescrStatsW(wt_df2, weights=wt_df['weight'])
        outfilenmw = outfilenm.replace('.csv', 'w.csv')
        wt_df.to_csv(outfilenmw, encoding='utf-8')
        
        wt_metrics_report = metrics.precision_recall_fscore_support(wt_df[handcode_2bin], wt_df.pred, sample_weight=wt_df.weight)

        prec_wt = wt_metrics_report[0][1]
        rec_wt = wt_metrics_report[1][1]
        f1_wt = wt_metrics_report[2][1]
        yes_wt = wt_metrics_report[3][1]
        no_wt = wt_metrics_report[3][0]
 
#    print('\nWeighted correlation',pbr_wt.corrcoef[0,1])
    print('\nWeighted precision =',prec_wt,'\nWeighted recall =',rec_wt,'\nWeighted F1 =',f1_wt,'\nWeighted N training cases NO =',no_wt,' YES =', yes_wt,' Total =',no_wt+yes_wt,'\nWeighted correlation',r_wt.corrcoef[0,1],'\nWeighted point-biserial correlation',pbr_wt.corrcoef[0,1])

    tt=tt+'_wt'
    return {'yes': yes_wt, 'no':no_wt, 'prec': prec_wt, 'rec': rec_wt, 'f1':f1_wt, 'pbr':pbr_wt.corrcoef[0,1], 'r':r_wt.corrcoef[0,1],'tt':tt, 'clfname':clfname}

def wt_metrics_leeann(handcode_2bin, handcode_full, model, clfname, weights, theme, cueold, tt):

    print('\n#### GETTING WEIGHTED METRICS')
    #metrics report 
#    metrics_report = metrics.precision_recall_fscore_support(y, df.pred)
    #note these won't be the same as F1 from step-down process in cross-validation because fold is randomly assigned
#    prec = metrics_report[0][1]
#    rec = metrics_report[1][1]
#    f1 = metrics_report[2][1]
#    yes = metrics_report[3][1]
#    no = metrics_report[3][0]
#    pbr = df['pred'].corr(df[handcode_full])

    if model=='log_reg':
        if tt=='test':
            outfile1nm = clfname.replace('.csv', '_pred-test.csv')
        if tt=='train':
            outfile2nm = clfname.replace('.csv', '_pred-train.csv')
        
        testdf = pd.read_csv(outfile1nm)
        traindf = pd.read_csv(outfile2nm)
        traintestdf = [testdf, traindf]
        df = pd.concat(traintestdf)
        
        wt_dfall = add_weights(df,weights,theme,cueold)
        print(wt_dfall.shape)
        #try making just a df for these two columns
        wt_df1 = wt_dfall[['pred',handcode_full]].copy()
        pbr_wt2 = sm.stats.DescrStatsW(wt_df1, weights=wt_dfall['weight'])
        wt_df2 = wt_dfall[['pred_proba',handcode_full]].copy()
        r_wt2 = sm.stats.DescrStatsW(wt_df2, weights=wt_dfall['weight'])
        outfilenmw = outfile2nm.replace('.csv', 'testw.csv')
        wt_dfall.to_csv(outfilenmw, encoding='utf-8')
        
        wt_metrics_report2 = metrics.precision_recall_fscore_support(wt_dfall[handcode_2bin], wt_dfall.pred, sample_weight=wt_dfall.weight)

        prec_wt2 = wt_metrics_report2[0][1]
        rec_wt2 = wt_metrics_report2[1][1]
        f1_wt2 = wt_metrics_report2[2][1]
        yes_wt2 = wt_metrics_report2[3][1]
        no_wt2 = wt_metrics_report2[3][0]
 
#    print('\nWeighted correlation',pbr_wt.corrcoef[0,1])
    print('\nWeighted precision =',prec_wt2,'\nWeighted recall =',rec_wt2,'\nWeighted F1 =',f1_wt2,'\nWeighted N training cases NO =',no_wt2,' YES =', yes_wt2,' Total =',no_wt2+yes_wt2,'\nWeighted correlation',r_wt2.corrcoef[0,1],'\nWeighted point-biserial correlation',pbr_wt2.corrcoef[0,1])

    tt=tt+'_wt'
    return {'yes': yes_wt2, 'no':no_wt2, 'prec': prec_wt2, 'rec': rec_wt2, 'f1':f1_wt2, 'pbr':pbr_wt2.corrcoef[0,1], 'r':r_wt2.corrcoef[0,1],'tt':tt, 'clfname':clfname}