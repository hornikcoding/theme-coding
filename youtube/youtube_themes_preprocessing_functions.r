


# a set of "words" to exclude from cleaned up text fields
stop_words <- function(){
  unique(c(
    "rt","the","a","i","https","t.co","my","to","??","?","in","and","was",
    "you","your","just","for","of","is","that","me","all","if","this","out",
    "she","her","he","him","i?","but","what","it","so","with","u","on","at",
    "being","across","up","as","one","be","amp","they","please","other","to?",
    "are","not","s","have","after","only","do","i'm","we","are","?","s","no",
    "an","when","we","im","has","from","who","got","will","because","you're",
    "m","s","re","can","too","ur","ya","about","his","ever","or","by","t","y?",
    "you?","w","us","c","ok","hey","lol","oi",
    as.character(0:9),letters,LETTERS, "\\|\\|\\|", "|||", 
    stopwords::data_stopwords_smart$en, stopwords::stopwords(),
    "um", "uh", "10", "http", "yeah", "bit.ly", 
    # additions july 2018
    "i>>"))
}


# remove urls, scrub with twitter prep function, then lemmatize 
# note to self: this is the cleaning routine for `scrub7` in earlier training 
preprocess_youtube_text <- function(text_vec, ...){
  require(magrittr)  # for pipe 
  text_vec %>% 
    remove_urls(remove_www=TRUE) %>% 
    preprocess_tweet_text(stop_words=stop_words(), ...) %>% 
    sapply(textstem::lemmatize_strings, USE.NAMES=FALSE)
}


# remove urls for cleaning text fields 
remove_urls <- function(text, remove_www=FALSE){ 
  text <- gsub("http[^ \n]+", "", text) 
  if (remove_www)
    text <- gsub("www[^ \n]+", "", text)
  return(text)
}

# remove stopwords for cleaning text fields 
remove_stops <- function(doc, stops){
  stopifnot(all(is.character(doc), length(doc)==1))
  doc_words <- unlist(strsplit(doc, split = " "))
  doc_words <- doc_words[!doc_words %in% c("", " ")]
  doc_words <- doc_words[!doc_words %in% stops]
  doc_reassembled <- paste(doc_words, collapse = " ")
  return(doc_reassembled)
}


# preprocessing function -- designed for tweets but works fine for youtube text
# (some twitter-specific steps are included for consistency, but have no effect)
preprocess_tweet_text <- function(text_vec, stop_words=NULL, to_lower=TRUE, 
                                  trim_ws=TRUE, to_utf8=TRUE){
  
  ### if `to_utf8`: convert to utf8 char encoding 
  # text_vec <- enc2utf8(text_vec) # dont use this -- iconv() is better 
  if (to_utf8)
    text_vec <- iconv(text_vec, "latin1", "UTF-8")
    
  
  # "word/word" and "word-word" and "word...word" should be "word word" 
  text_vec <- gsub("\\/|\\-|\\.{2,}", " ", text_vec)
  
  # eliminate multibyte strings and char encoding problems caused by windows
  text_vec <- textclean::replace_non_ascii(text_vec, replacement="", 
                                           remove.nonconverted=TRUE)
  text_vec <- gsub("i>>", "", text_vec)
  
  ### remove urls 
  text_vec <- gsub("htt[^ ]*", "", text_vec)
  
  ### remove hashtag symbol
  text_vec <- gsub("#", "", text_vec)
  
  ### remove user mention symbol 
  text_vec <- gsub("@", "", text_vec)
  
  ### remove "RT " prefix when it is string-initial 
  text_vec <- gsub("^RT ", "", text_vec, ignore.case=TRUE)
  
  ### replace numerals related to themes with spelled-out equivalents 
  text_vec <- gsub("\\b18\\b", " eighteen ", text_vec)
  text_vec <- gsub("\\b19\\b", " nineteen ", text_vec)
  text_vec <- gsub("\\b21\\b", " twentyone ", text_vec)
  
  ### remove digits and punctuation 
  text_vec <- gsub("[[:punct:]]|\\d", "", text_vec)
  
  ### remove stopwords (if not tagging/parsing/interpreting) 
  if (!is.null(stop_words))
    text_vec <- sapply(text_vec, remove_stops, stops=stop_words,USE.NAMES=FALSE)
  
  ### if `to_lower`: lowercase 
  if (to_lower)
    text_vec <- tolower(text_vec)
  
  ### if `trim_ws`: trim whitespace 
  if (trim_ws)
    text_vec <- trimws(gsub(" +", " ", text_vec), which="both")
  
  
  return(text_vec) 
}


