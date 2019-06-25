# Feature Extration

## Links
1. [scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html)
2. [Ultimate guide to deal with Text Data (using Python)](https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/)
2. [Stanford CS224N: NLP with Deep Learning](http://onlinehub.stanford.edu/cs224)
## Basic feature extraction using text data
1. Number of words
2. Number of characters
3. Average word length
4. Number of stopwords
5. Number of special characters
6. Number of numerics
7. Number of uppercase words
## Basic Text Pre-processing of text data
1. Lower casing
2. Punctuation removal
3. Stopwords removal
4. Frequent words removal
5. Rare words removal
6. Spelling correction
7. Tokenization
8. Stemming
9. Lemmatization
## Advance Text Processing
1. N-grams
2. Term Frequency
3. Inverse Document Frequency
4. Term Frequency-Inverse Document Frequency (TF-IDF)
5. Bag of Words
6. Sentiment Analysis
7. Word Embedding





# Relevant Packages

## NLTK
A basic NLP library in python.
[Natural Language Toolkit](https://www.nltk.org)

#### WordNet
A thesaurus containing lists of synonym sets and hypernyms (“is a” relationships). 
 
e.g. synonym sets containing “good”:
```python
from nltk.corpus import wordnet as wn
poses = { 'n':'noun', 'v':'verb', 's':'adj (s)', 'a':'adj', 'r':'adv'}
for synset in wn.synsets("good"):
	print("{}: {}".format(poses[synset.pos()],
		", ".join([l.name() for l in synset.lemmas()])))
```
e.g. hypernyms of “panda”:
```python
from nltk.corpus import wordnet as wn
panda = wn.synset("panda.n.01")
hyper = lambda s: s.hypernyms()
list(panda.closure(hyper))
```

#### stopwords 

e.g. removal of stop words
```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
````

#### Stemming 

e.g. the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. 
````python
from nltk.stem import PorterStemmer
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
````

## Gensim 
[gensim – Topic Modelling in Python](https://radimrehurek.com/gensim/)



## TextBlob
[TextBlob: Simplified Text Processing](https://textblob.readthedocs.io/en/dev/)

