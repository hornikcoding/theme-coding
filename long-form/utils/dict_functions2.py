# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:17:20 2017

Copy of dict functions file with correct path for Leeann's computer

Additional functions needed for dictionaries...
Created on Tue Feb 28 10:46:31 2017

@author: lgibson
"""

from sklearn import metrics

import csv
import re
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm

sys.path.append('//FILE3/TobaccoSurveyData/P50 - Youth Tobacco Tracking/2_Content Analysis/Data/_Python/Ecig-themes/scripts')


from utils import class_functions2 as cf

def clean_df(docs, scode, scue, npm=False, pm=False, nobtn=False):
    '''
    pre-process data and limit to the dataset you want
    '''

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
    docs, y = cf.limit(docs, scode)
        
    return docs, y

def load_regexes(filename):
    '''
    load regex and compile into dictionary
    '''
    
    with open(filename) as f:
        regex_dict = dict([(r['label'],re.compile('(%s)' % r['regex'], re.I)) for r in csv.DictReader(f)])    
    
    return regex_dict    

def highlight(text, feats, highlight_type="text"):
    '''
    given a text and a set of compiled regex features
    highlight all hits by transforming to upper case
    also highlights all TM words (helps with the misses) - must be included in every regex file
    '''
    pre = '<span class="hl">' if highlight_type=='html' else '__'
    post = '</span>' if highlight_type=='html' else '__'
    pre2 = '<b>' if highlight_type=='html' else '_#_'
    post2 = '</b>' if highlight_type=='html' else '_#_'

    for label,feat in feats.items():
        if label!='tm':
            text = feat.sub(lambda x: '%s%s%s' % (pre,x.group(1).upper(),post), text)
        else:    
            text = feat.sub(lambda x: '%s%s%s' % (pre2,x.group(1).upper(),post2), text)
  
    return text

def score_text(text, feats):
    '''
    create a list of counts for each of supplied
    regex feats in text
    '''
    
    counts = {}
    for label,feat in feats.items():
        counts[label] = len(feat.findall(text))
    return counts

def test_logic_tobyt(hit_dict):
    if hit_dict['tobacco_maybe']>0:
        return 1
    else:
        return 0

def test_logic_tob(hit_dict):
    '''
        tobacco relevant if:
            - if tobacco_maybe > 0
                - for each pair check
                   - _n > 0 ?
                       - yes _tm > 0 => TRUE = tobacco rel - FALSE = go to next pair
    '''
    
    pairs = set([ re.sub('_n|_tm','',label) for label in hit_dict if label.endswith('_n') or label.endswith('_tm')]) 
    
    if hit_dict['tobacco_maybe']>0:
        result = []
        for pair in pairs:
            pair_n = pair+'_n'
            pair_tm = pair+'_tm'

            if hit_dict[pair_n] > 0:
                if hit_dict[pair_tm]>0:
                    return 1
                
            result.append(hit_dict[pair_n])
        if sum(result)>0:
            return 0
        else:
            return 1
                    
    else:
        return 0

def test_logic_addict(hit_dict):
    '''
    #version 1
    if hit_dict['addict_before20_after20'] > 0 or hit_dict['habit_before5_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    #versions 2-3
    if hit_dict['addict_before20_after20'] > 0 or hit_dict['habit_before5_after5'] > 0 or hit_dict['crave_before5_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    #version 5,6:
    if hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0:
        return 1
    else:
        return 0
    #version 4 & 7 on:
    if hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    '''
    #version 12:
    if hit_dict['habit_before10'] > 0 or hit_dict['habit_after10'] > 0 or hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    '''
    #version 13:
    if hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    #version 14:
    if hit_dict['habit_before10'] > 0 or hit_dict['habit_after10'] > 0 or hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    #version 15:
    if hit_dict['habit_bef_quit'] > 0 or hit_dict['habit_aft_quit'] > 0 or hit_dict['habit_before10'] > 0 or hit_dict['habit_after10'] > 0 or hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    #version 16 on:
    if hit_dict['pack_bef_day'] > 0 or hit_dict['habit_bef_quit'] > 0 or hit_dict['habit_aft_quit'] > 0 or hit_dict['habit_before10'] > 0 or hit_dict['habit_after10'] > 0 or hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0    
    #version 18 on:
    if hit_dict['urge_bef_smk'] > 0 or hit_dict['urge_aft_smk'] > 0 or hit_dict['pack_bef_day'] > 0 or hit_dict['habit_bef_quit'] > 0 or hit_dict['habit_aft_quit'] > 0 or hit_dict['habit_before10'] > 0 or hit_dict['habit_after10'] > 0 or hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0     
    #version 21:
    if hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    #version 22:
    if hit_dict['pack_bef_day'] > 0 or hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    #version 23:
    if hit_dict['habit_before10'] > 0 or hit_dict['habit_after10'] > 0 or hit_dict['pack_bef_day'] > 0 or hit_dict['addict_before20'] > 0 or hit_dict['addict_after20'] > 0 or hit_dict['habit_before5'] > 0 or hit_dict['habit_after5'] > 0 or hit_dict['crave_before5'] > 0 or hit_dict['crave_after5'] > 0 or hit_dict['get_before2'] > 0 or hit_dict['hook_before5_smok?'] > 0 or hit_dict['chain_smok?'] > 0 or hit_dict['chain-smok?'] > 0:
        return 1
    else:
        return 0
    '''
    
def test_logic_health2(hit_dict):
    
    if hit_dict['health1'] > 0 or hit_dict['health2'] > 0:
        if hit_dict['2screen'] > 2 or hit_dict['1screen'] > 1 or hit_dict['0screen'] > 0:
            return 0
        else:
            return 1
        
    else:
        return 0
    
def test_logic_policy(hit_dict):
    '''
    #v1
    if hit_dict['policy1'] > 0 or hit_dict['policy2'] > 0 or hit_dict['policy3'] > 0:
        return 1
    else:
        return 0
    #v2
    if hit_dict['policy1'] > 0 or hit_dict['policy2'] > 0 or hit_dict['policy3'] > 0 or hit_dict['policy4'] > 0:
        return 1
    else:
        return 0
    #v3-5
    if hit_dict['policy1'] > 0 or hit_dict['policy2'] > 0 or hit_dict['policy3'] > 0 or hit_dict['policy4'] > 0 or hit_dict['policy5'] > 0:
        return 1
    else:
        return 0
    '''
    #v6
    if hit_dict['policy1'] > 0 or hit_dict['policy2'] > 0 or hit_dict['policy3'] > 0 or hit_dict['policy4'] > 0 or hit_dict['policy5'] > 0 or hit_dict['policy6'] > 0:
        return 1
    else:
        return 0
    
def test_logic_youth(hit_dict):

    '''
       youth theme if:
               - if youth > 0
   '''

#    if hit_dict['ryouth_1'] > 0:
#        return 1
    '''
    #used to use this - but losing too much with 1 hit for exclusions
    if hit_dict['ryouth_1'] > 0 or hit_dict['ryouth_2'] > 0 or hit_dict['ryouth_1hook'] > 0 or hit_dict['ryouth_2hook'] > 0 or hit_dict['age1'] > 0 or hit_dict['age2'] > 0:
        if hit_dict['not_ysmk'] > 0 or hit_dict['shs'] > 0 or hit_dict['health'] > 0:
            return 0
        elif hit_dict['mj'] > 0:
            if hit_dict['ryouth_1tob'] > 0 or hit_dict['ryouth_2tob'] > 0:
                return 1
            else:
                return 0
        else:
            return 1
    else:
        return 0
    #used with version 18
    if hit_dict['ryouth_1'] > 0 or hit_dict['ryouth_2'] > 0 or hit_dict['ryouth_1hook'] > 0 or hit_dict['ryouth_2hook'] > 0 or hit_dict['age1'] > 0 or hit_dict['age2'] > 0:
        if hit_dict['not_ysmk'] > 2 or hit_dict['shs'] > 2 or hit_dict['health'] > 2:
            return 0
        elif hit_dict['mj'] > 0:
            if hit_dict['ryouth_1tob'] > 0 or hit_dict['ryouth_2tob'] > 0:
                return 1
            else:
                return 0
        else:
            return 1
    else:
        return 0
    #used version 19 on
    if hit_dict['rschool']>0 or hit_dict['ryouth_1'] > 0 or hit_dict['ryouth_2'] > 0 or hit_dict['ryouth_1hook'] > 0 or hit_dict['ryouth_2hook'] > 0 or hit_dict['age1'] > 0 or hit_dict['age2'] > 0:
        if hit_dict['not_ysmk'] > 2 or hit_dict['shs'] > 2 or hit_dict['health'] > 2:
            return 0
        elif hit_dict['mj'] > 0:
            if hit_dict['ryouth_1tob'] > 0 or hit_dict['ryouth_2tob'] > 0:
                return 1
            else:
                return 0
        else:
            return 1
    else:
        return 0
    '''
    #used version 25 on
    if hit_dict['rschool']>0 or hit_dict['ryouth_1'] > 0 or hit_dict['ryouth_2'] > 0 or hit_dict['ryouth_1hook'] > 0 or hit_dict['ryouth_2hook'] > 0 or hit_dict['age1'] > 0 or hit_dict['age2'] > 0:
        if hit_dict['not_ysmk'] > 1 or hit_dict['shs'] > 1 or hit_dict['health'] > 1:
            return 0
        elif hit_dict['mj'] > 0:
            if hit_dict['ryouth_1tob'] > 0 or hit_dict['ryouth_2tob'] > 0:
                return 1
            else:
                return 0
        else:
            return 1
    else:
        return 0

def dict_code(docs, regexes, test_logic):
    texts = docs['new_text'].str.lower()
    results = []
    for r in texts:
        feats = score_text(r, regexes)
        pred = test_logic(feats) 
        results.append(pred)
    return results

def tagged_text(docs,regexes, CELL_MAX=300000):
    docs['new_text']=docs['new_text'].str.lower()
    texts = docs.to_dict('records')
    results = []
    for r in texts:
        feats = score_text(r['new_text'], regexes)
        feats2 = dict([('kw_%s' % k , v ) for k,v in feats.items()])
        htext = highlight(r['new_text'],regexes, highlight_type='html')
        feats2.update({'tagged_text': htext[:CELL_MAX]})
        r.update(feats2)
        results.append(r)
        sys.stdout.write('.'); sys.stdout.flush();  # print a small progress bar
    return results

def metrics_bob(docs, y, handcode_full, kname, tt):
    '''
    Get metrics for the classifier, save probabilities to .csv file
    Return: N Yes, N No, N Total, Precision, Recall
    '''
    print('\n#### GETTING METRICS')
    #metrics report 
    metrics_report = metrics.precision_recall_fscore_support(y, docs.pred_dict)
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
    pbr = docs['pred_dict'].corr(docs[handcode_full])

    dfout = docs.copy()
    del dfout['new_text']
    del dfout['ArticleContent']

    if tt=='train':
        outfilenm = kname.replace('.csv', '_pred-train.csv')
    elif tt=='test':
        outfilenm = kname.replace('.csv', '_pred-test.csv')
    elif tt=='test-all':
        outfilenm = kname.replace('.csv', '_pred-test-all.csv')
    dfout.to_csv(outfilenm, encoding='utf-8', index=False)

    print('\nPrecision for NO YES =',prec,'\nRecall for NO YES =',rec,'\nF1 for NO YES =',f1,'\nN training cases NO =',no,' YES =', yes,' Total =',no+yes,'\nPoint-biserial correlation',pbr)

    return {'yes': yes, 'no':no, 'prec': prec, 'rec': rec, 'f1':f1, 'pbr':pbr, 'tt':tt, 'kname':kname}

def wt_metrics_bob(handcode_2bin, handcode_full, kname, weights, theme, cueold, tt):

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

    if tt=='test':
        outfilenm = kname.replace('.csv', '_pred-test.csv')
    elif tt=='test-all':
        outfilenm = kname.replace('.csv', '_pred-test-all.csv')
        
    df = pd.read_csv(outfilenm)
    wt_df = cf.add_weights(df,weights,theme,cueold)
    print(wt_df.shape)
    #try making just a df for these two columns
    wt_df1 = wt_df[['pred_dict',handcode_full]].copy()
    pbr_wt = sm.stats.DescrStatsW(wt_df1, weights=wt_df['weight'])
    outfilenmw = outfilenm.replace('.csv', 'w.csv')
    wt_df.to_csv(outfilenmw, encoding='utf-8', index=False)

    wt_metrics_report = metrics.precision_recall_fscore_support(wt_df[handcode_2bin], wt_df.pred_dict, sample_weight=wt_df.weight)

    prec_wt = wt_metrics_report[0][1]
    rec_wt = wt_metrics_report[1][1]
    f1_wt = wt_metrics_report[2][1]
    yes_wt = wt_metrics_report[3][1]
    no_wt = wt_metrics_report[3][0]
 
#    print('\nWeighted correlation',pbr_wt.corrcoef[0,1])
    print('\nWeighted precision =',prec_wt,'\nWeighted recall =',rec_wt,'\nWeighted F1 =',f1_wt,'\nWeighted N training cases NO =',no_wt,' YES =', yes_wt,' Total =',no_wt+yes_wt,'\nWeighted point-biserial correlation',pbr_wt.corrcoef[0,1])

    tt=tt+'_wt'
    return {'yes': yes_wt, 'no':no_wt, 'prec': prec_wt, 'rec': rec_wt, 'f1':f1_wt, 'pbr':pbr_wt.corrcoef[0,1], 'tt':tt, 'kname':kname}

def create_HTML(filename, data):
    '''
    Create an HTML report for HITS or MISSES to make viewing easier
    '''
    
    doc_html = "<div id=\"{num}\"><h3>{num}. {title}</h3><div>{hit_table}</div><div>{text}</div></div>"
    
    #Stella's
    #reportfile='H:/13. Research/TCORS/Policy theme keywords/Policy keywords Jan 2016/Reports/report_2.html'
    reportfile = '//FILE3/TobaccoSurveyData/P50 - Youth Tobacco Tracking/2_Content Analysis/Data/_Python/Ecig-themes/scripts/report_2.html'

    template = open(reportfile,'r').read()
    
    docs = []
    for row in data:
        
        hit_html = []
        
        kw_items = [item for item in row if item.startswith('kw')]
        kw_items.sort()
        
        for item in kw_items: 
            hit_html.append('<tr><td>%s</td><td>%i</td></tr>' % (item.replace('kw_',''), row[item]))

        hit_table = '<table><tbody>%s</tbody></table>' % ('\n'.join(hit_html))
        docs.append(doc_html.format(num=row['ArticleID'], title=row['SourceTitle'], hit_table=hit_table, text=row['tagged_text']))
        
    docs_html = '\n'.join(docs)    
        
    with open(filename,'w', encoding='utf-8') as out:
        out.write(template % docs_html)