# Feature Extration

## Links
1. [scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html)
2. [Ultimate guide to deal with Text Data (using Python)](https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/)
2. [Stanford CS224N: NLP with Deep Learning](http://onlinehub.stanford.edu/cs224)
4. [Traditional Methods for Text Data](https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41)
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

## scikit-learn
[scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html)

#### Bag of Words

e.g. unigrams (n=1)
````python
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['tweet'])
train_bow
> > 31962x1000 sparse matrix of type '<class 'numpy.int64'>'
	with 128380 stored elements in Compressed Sparse Row format>
````

#### TF-IDF

Term frequency is simply the ratio of the count of a word present in a document, to the length of the document.

Inverse document frequency (IDF) is the log of the ratio of the total number of documents to the number of documents in which that word is present. The more the value of IDF, the more unique is the word.

TF-IDF is the multiplication of the TF and IDF.

e.g. implementation in sklearn:

````python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['tweet'])

train_vect
> > 31962x1000 sparse matrix of type '<class 'numpy.float64'>'
	with 114033 stored elements in Compressed Sparse Row format>
````


## TextBlob
[TextBlob: Simplified Text Processing](https://textblob.readthedocs.io/en/dev/)

#### Spelling correction

Spelling correction is a useful pre-processing step because this will help us in reducing multiple copies of words.
```python
from textblob import TextBlob
train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))
```

#### Tokenization

Tokenization refers to dividing the text into a sequence of words or sentences. In the below example, we have used the textblob library to first transform our tweets into a blob and then converted them into a series of words.

```python
TextBlob(train['tweet'][1]).words
> > WordList(['thanks', 'lyft', 'credit', 'cant', 'use', 'cause', 'dont', 'offer', 'wheelchair', 'vans', 'pdx', 'disapointed', 'getthanked'])
```

#### Lemmatization

Lemmatization is a more effective option than stemming because it converts the word into its root word, rather than just stripping the suffices. It makes use of the vocabulary and does a morphological analysis to obtain the root word.

```python
from textblob import Word
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
```

#### N-grams

N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used.

e.g. extract bigrams from tweets using the ngrams function of the textblob library.
````python
from textblob import TextBlob
TextBlob(train['tweet'][0]).ngrams(2)
````
#### Sentiment Analysis

e.g. detecting the sentiment using the textblob library, it returns a tuple representing polarity and subjectivity of each tweet.
```python
train['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)
0    (-0.3, 0.5354166666666667)
1                    (0.2, 0.2)
2                    (0.0, 0.0)
3                    (0.0, 0.0)
4                    (0.0, 0.0)
Name: tweet, dtype: object
```

e.g. only extract polarity as it indicates the sentiment as value nearer to 1 means a positive sentiment and values nearer to -1 means a negative sentiment. This can also work as a feature for building a machine learning model.

```python
train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
train[['tweet','sentiment']].head()
```


## Gensim 
[gensim – Topic Modelling in Python](https://radimrehurek.com/gensim/)


