import re
import os


### this file defines some functions and a class used to classify tweets as 
### relevant to themes 'addiction', 'health', 'policy', and 'youth'. 
### 
### lists of inclusionary, exclusionary, etc. terms are loaded when the 
### `ThemeClf()` class is instantiated with a theme. 
### 
### see script `example_theme_classification.py` for example usage



# function to get a theme's keywords, exclusions, or 'not enough' words 
# term type should be: 'exclude', 'keyword', or 'notenough'
def get_termlist(theme, term_type, term_loc):

  terms = []

  for line in open(os.path.join(term_loc, term_type + '_' + theme + '.txt')):
    if not line.strip().startswith('#'):
      terms.append(line.strip())

  return(terms)




# function to count the number of keyword matches in a text field
def count_matches(term_list, text_field):

  if len(term_list) == 0:
    return 0
  else: 
    term_regex = r'\b' + (r'\b|\b'.join(term_list)) + r'\b'
    return len(re.findall(term_regex, text_field))





# theme classifier class (instantiate w a theme string + path to .txt files) 
class ThemeClf(object):

  # on instantiate, get the keywords, exclusion terms, and not-enough terms 
  def __init__(self, theme, term_loc):

    self.theme = theme
    self.keywords = get_termlist(theme, 'keyword', term_loc)
    self.exclude = get_termlist(theme, 'exclude', term_loc)
    self.not_enough = get_termlist(theme, 'notenough', term_loc)
    # god terms are optional -- see 'youth' example for usage
    self.god_terms = []


  # takes in a tweet, returns a (boolean) label 
  def classify_tweet(self, tweet):

    tweet = tweet.lower()

    # if the theme has any god terms, then return `True` if one appears 
    if len(self.god_terms) > 0:
      if count_matches(self.god_terms, tweet) > 0:
        return True

    # if there are any exclusion terms, return `False` 
    if count_matches(self.exclude, tweet) > 0:
      return False

    # count the number of hits in the tweet 
    n_hits = count_matches(self.keywords, tweet)

    # if there are no hits, return `False` 
    if n_hits == 0:
      return False
    elif n_hits > 1:
      # if there are multiple hits, return `True`
      return True
    else:
      # if there's only one hit, then... 
      # return `True` if there's *no* not-enough terms, and `False` otherwise
      return count_matches(self.not_enough, tweet) == 0




