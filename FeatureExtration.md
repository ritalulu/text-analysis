# Feature Extration

## Links
1. [scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html)
2. [Stanford CS224N: NLP with Deep Learning](http://onlinehub.stanford.edu/cs224)


## Vectorizer 
1. using hashing trick
2. 
We call vectorization the general process of turning a collection of text documents into numerical feature vectors. 

### Bag of Words

The specific strategy (tokenization, counting and normalization) is called the **Bag of Words** or “Bag of n-grams” representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.

Sparsity: 
>As most documents will typically use a very small subset of the words used in the corpus, the resulting matrix will have many feature values that are zeros (typically more than 99% of them).

Using stop words. 
>You should also make sure that the stop word list has had the same preprocessing and tokenization applied as the one used in the vectorizer. The word we’ve is split into we and ve by CountVectorizer’s default tokenizer, so if we’ve is in stop_words, but ve is not, ve will be retained from we’ve in transformed text.

### Tf–idf term weighting¶


## WordNet
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
## Gensim 
