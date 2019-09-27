
# Youtube Themes `README.md`

This is the repository for the Youtube themes classification task for the UPenn social media tobacco/ecig project. 

The code in this repository performs the following steps:

1. prepare Youtube videos datasets for modeling ([`youtube_themes-preprocess.rmd`](youtube_themes-preprocess.rmd))
2. apply a classifier for each theme, assigning predictions to new videos ([`youtube_themes_classify.py`](youtube_themes_classify.py)); 

Toy video dataset example is included (``"youtube_videos-toy-clean.csv"`) in the directory `"data/motherload/"`. You could include captions data and video data in separate files -- but these have already been combined in the example. This is where you would substitute your data, either by placing a copy of the files there, or else changing all path references to `"data/motherload/"` to the location of these files on the computer you're working from. 

All package dependencies for R are listed in the relevant `.rmd` files. All Python code is written in 3.6, and requires the packages `numpy`, `pandas`, and `scikit-learn`. 



### Step 1: Prepare Youtube data for modeling

###### Input/Output files

Input files:

- `"data/motherload/youtube_videos-toy-clean.csv"`

Output files:

- `"data/prepped/youtube_videos-toy-clean-prepped.csv"`


###### Description of main file: [`youtube_themes-preprocess.rmd`](youtube_themes-preprocess.rmd)

The notebook [`youtube_themes-preprocess.rmd`](youtube_themes-preprocess.rmd) creates a preprocessed text field used (in Step 2) as the input features for prediction. This step cleans up the Youtube "motherload" data, concatenates its text fields, and preprocesses its text. 

These are the main steps in the notebook: 

- Prepare collection of relevant Youtube videos for classification: 
    - Join videos with captions, format and check dates (for daily counts in Step 4); 
    - Concatenate text fields (creating `text_blob`); and 
    - Preprocess text blob (creating `text_blob_scrubbed`) and write to file. 

The text preprocessing routine consists of the following (applied to each video): 

- Concatenate all text fields (`title`, `channel_title`, `description`, `tags`, `caption`) to create `text_blob`; 
- Apply the function `preprocess_youtube_text()` from file [`youtube_themes_preprocessing_functions.r`](youtube_themes_preprocessing_functions.r) to `text_blob`. to create the field `text_blob_scrubbed`. The following transformations are applied:
    - urls are removed; 
    - slashes, hyphens, and ellipses are converted to spaces; 
    - text is lowercased and non-ascii characters are removed; 
    - a list of stopwords are removed; 
    - text is lemmatized (morphological variants are converted to a common form).



### Step 2: Train theme classifiers and assign predictions

###### Input/Output files

Input files:

- `"data/prepped/youtube_videos-toy-clean-prepped.csv"`
- `"youtube_themes_hypers.json"`

Output files:

- `"data/output/pred_*.csv"` for `*` in `['addiction','health','policy','youth']` 



###### Description of main file: [`youtube_themes_classify.py`](youtube_themes_classify.py)

The script [`youtube_themes_classify.py`](youtube_themes_classify.py) trains an ensemble classifier for each theme, and generates theme-relevance predictions on both the evaluation data and the large collection of videos to be used for deriving daily counts. All predictions are saved to intermediate `.csv` files containing just the video ids and model predictions. The files written by this script are used as the inputs to the evaluation notebook in Step 3. 

Here is an outline of what happens when you execute the script `youtube_themes_classify.py`: 

- Import functions from [`youtube_themes.py`](youtube_themes.py), which implement the classification algorithm described below. 
- Read in files to be used for classification or prediction (see list above).
- For each theme, train binary classifiers predicting the training labels from unigram features constructed from the `"text_blob_scrubbed"` column of the training dataset (1498 rows): 
    - two decision tree classifiers (one with count features, another with tf-idf weighting); 
    - two stochastic gradient descent classifiers (one with count features, another with tf-idf weighting); and 
    - an additional classifier that returns a `True` prediction if the input text contains any of a given set of keywords associated with each theme (see `keywords` dict, defined around line 50).
- For each theme, call the function `youtube_clf()` to apply all five classifiers to each video in the evaluation dataset and the motherload dataset (creating boolean "sub-predictions"). 
- For each theme and each video, call `subpreds_to_preds()` to aggregate the five sub-predictions and derive a final prediction. 
    - Note that the `threshold` parameter of `subpreds_to_preds()` can be treated as a tunable hyper-parameter (the number of submodels that must predict `True` for the final prediction to be `True`). I have found the value of `3` to be optimal in this case. 
