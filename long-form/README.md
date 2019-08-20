This folder contains our long-form classifier pickle files, a script, and a toy test set.

It’s pieced together from other code, so there is some duplication of cleaning the texts, but I don’t think that is a problem. The toy data set comes back with the expected data.

When you open code_clf_themes_loop_forNORC.py – you should only have to change the datafile name and the path (listed twice at the top). You will get 4 files out (one for each theme, ending in _probas) that give the probabilities for each text (this is because we had different texts handcoded for each theme).

The text file needs to have ArticleID, ArticleTitle, and ArticleContent. ArticleTitle and ArticleContent are concatenated before classification in this code. Then we find a window around the relevant keywords. You can just leave ArticleTitle blank if you don’t have 2 fields for the content data. You can also toggle whether or not ArticleTitle is included by changing title=True to title=False.

We do get these 2 warnings, but they don’t seem to cause any trouble:
•	sklearn\base.py:311: UserWarning: Trying to unpickle estimator CountVectorizer from version 0.18.1 when using version 0.19.1. This might lead to breaking code or invalid results. Use at your own risk. UserWarning)
•	sklearn\base.py:311: UserWarning: Trying to unpickle estimator LogisticRegression from version 0.18.1 when using version 0.19.1. This might lead to breaking code or invalid results. Use at your own risk. UserWarning)

This was built with Python 3.5. 
