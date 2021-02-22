
import numpy as np
import nltk 
from typing import List 
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup #for parsing XML
from benchmarker import Benchmarker

#functions
def build_vocab_and_token_list(reviews: List[str], vocab_map: List[str], vocab_map_index: int) -> List[str]:
    '''Takes the input strings of reviews and a total vocabulary. Populates the vocab map and returns a list of tokens from the input string'''
    token_list = []
    for each in reviews:
        tokens = tokenize(each.text)
        token_list.append(tokens)
        for token in tokens:
            if token not in vocab_map:
                vocab_map[token] = vocab_map_index
                vocab_map_index += 1
    return token_list 

def tokenize(s: str) -> List[str]:
    '''custom tokenizer that converts review string to a list of tokens. '''
    s = s.lower() 
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if t not in stopwords] #remove known unhelpful words (List comprehension)
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]  #convert related words to same base
    tokens = [t for t in tokens if (len(t) > 2)] #remove short words 
    return tokens

def tokens_to_vector(tokens: List[str], label: int) -> List[int]:
    '''Create data-vectors from a list of tokens'''
    normalized_count = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        normalized_count[i] += 1
    normalized_count = normalized_count/normalized_count.sum() #normalizing data
    normalized_count[-1] = label
    return normalized_count

#convert different versions of a word to the same word, e.g. Dog, dogs, doggies = dog
wordnet_lemmatizer = WordNetLemmatizer() 

# get list of words that have no predictive value (provided in file)
# http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.strip() for w in open('stopwords.txt')) 

#dataset from http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
#get review strings from XML files--we do this twice, so it might be worth putting into a function, if we intend to reuse it.
positive_reviews = BeautifulSoup(open('electronics/positive.review').read(), features="html.parser")
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read(), features="html.parser")
negative_reviews = negative_reviews.findAll('review_text')

#Because there are more positive reviews that negative reviews, we will shuffle then truncate the positive reviews to have matching data sets.
#If we were going to do this repeatedly, we should get the lengths of positive and negative reviews, and shuffle and truncate whichever is longer.
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)] #match qt of positive and negative reviews

#dictionary of vocabulary
word_index_map = {}
current_index = 0

#tokenized reviews by label
positive_tokenized = build_vocab_and_token_list(positive_reviews, word_index_map, current_index)
negative_tokenized = build_vocab_and_token_list(negative_reviews, word_index_map, current_index)

N = len(positive_tokenized) + len(negative_tokenized) #total number of reviews

#initialize a 2D matrix of size N x word_index_map length (plus one for the label)
data = np.zeros((N, len(word_index_map)+ 1)) 

i = 0
#populate row of data for each review.  Last column is label (1 for positive reviews)
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1) 
    data[i,:] = xy
    i+= 1

#populate row of data for each review.  Last column is label (0 for negative reviews)
for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i+= 1

np.random.shuffle(data)

X = data[:, :-1] #everything except the last columns (independent variables)
Y = data[:,-1] #only the last columns (dependent variable)

bench = Benchmarker()
bench.benchmark(X, Y)