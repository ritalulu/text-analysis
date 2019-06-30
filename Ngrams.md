# N-grams

## Links
1. [Sentiment Analysis with bag-of-words](http://ataspinar.com/2016/01/21/sentiment-analysis-with-bag-of-words/)
2. [Sentiment Analysis with bag-of-words (part 2)](https://ataspinar.wordpress.com/2016/02/01/sentiment-analysis-with-bag-of-words-part-2/)

## Unigram model

> In unigram model you only take individual words into account and give each word a specific subjectivity score. This subjectivity score can be looked up in a sentiment lexicon[1]. If the total score is negative the text will be classified as negative and if its positive the text will be classified as positive. It is simple to make, but is less accurate because it does not take the word order or grammar into account.

#### construct a sentiment lexicon

We can use one of the sentiment lexicons given in [1], but we dont really know in which circumstances and for which purposes these lexicons are created. Furthermore, in most of these lexicons the words are classified in a binary way (either positive or negative ). Bing Liu’s sentiment lexicon for example contains a list of a few thousands positive and a few thousand negative words.

Bo Pang and Lillian Lee used words which were chosen by two student as positive and negative words.
It would be better if we determine the subjectivity score of each word using some simple statistics of the training set. To do this we need to determine the class probability of each word present in the bag-of-words. This can be done by using pandas dataframe as a datacontainer (but can just as easily be done with dictionaries or other data structures). The code for this looks like:

```python
from sets import Set
import pandas as pd

BOW_df = pd.DataFrame(0, columns=scores, index='')
words_set = Set()
for review in training_set:
    score = review['score']
    text = review['review_text']
    splitted_text = split_text(text)
    for word in splitted_text:
        if word not in words_set:
            words_set.add(word)
            BOW_df.loc[word] = [0,0,0,0,0]
            BOW_df.ix[word][score] += 1
        else:
            BOW_df.ix[word][score] += 1
 ```
 
 Here split_text is the method for splitting a text into a list of individual words:
 
 ```python
 def expand_around_chars(text, characters):
    for char in characters:
        text = text.replace(char, &amp;quot; &amp;quot;+char+&amp;quot; &amp;quot;)
    return text

def split_text(text):
    text = strip_quotations_newline(text)
    text = expand_around_chars(text, '&amp;quot;.,()[]{}:;')
    splitted_text = text.split(&amp;quot; &amp;quot;)
    cleaned_text = [x for x in splitted_text if len(x)&amp;gt;1]
    text_lowercase = [x.lower() for x in cleaned_text]
    return text_lowercase
```
This gives us a DataFrame containing of the number of occurances of each word in each class:

```python
              Unnamed: 0     1     2      3      4      5
0                      i  4867  5092   9178  14180  17945
1                through   210   232    414    549    627
2                    all   499   537    923   1355   1791
3              drawn-out     1     0      1      1      0
4                      ,  4227  4779   8750  15069  18334
5               detailed     3     7     15     30     36
...                  ...   ...   ...    ...    ...    ...
31800           a+++++++     0     0      0      0      1
31801          nailbiter     0     0      0      0      1
31802            melinda     0     0      0      0      1
31803         reccomend!     0     0      0      0      1
31804         suspense!!     0     0      0      0      1
 
[31804 rows x 6 columns]
```
As we can see there are also quiet a few words which only occur one time. These words will have a class probability of 100% for the class they are occuring in.
This distribution however, does not approximate the real class distribution of that word at all. It is therefore good to define some ‘occurence cut off value’; words which occur less than this value are not taken into account.

By dividing each element of each row by the sum of the elements of that row we will get a DataFrame containing the relative occurences of each word in each class, i.e. a DataFrame with the class probabilities of each word. After this is done, the words with the highest probability in class 1 can be taken as negative words and words with the highest probability in class 5 can be taken as positive words.

We can construct such a sentiment lexicon from the training set and use it to measure the subjectivity of reviews in the test set. Depending on the size of the training set, the sentiment lexicon becomes more accurate for prediciton.

## Improving the bag-of-words with n-gram features

The biggest reason why bigram or trigram features are not used more often is that the number of possible combinations of words increases exponentially with the number of words. Theoretically, a document with 2.000 words can have 2.000 possible unigram features, 40.000 possible bigram features and 8.000.000.000 possible trigram features.

However, if we consider this problem from a pragmatic point of view we can say that most of the combinations of words which can be made, are grammatically not possible, or do not occur with a significant amount and hence don’t need to be taken into account.

Actually, we only need to define a small set of words (prepisitions, conjunctions, interjections etc) of which we know it changes the meaning of the words following it and/or the rest of the sentence.

Some examples of such words are:

![alt text](https://github.com/ritalulu/text-analysis/blob/master/bigram_words.png "bigram_words")

> There are a few conditions this “generate_ngrams” function needs to fulfill:
> 1. When it iterates through the splitted text and encounters a ngram-word, it needs to concatenate this word with the next word. So [“I”,”do”,”not”,”recommend”,”this”,”book”] needs to become [“I”, “do”, “not recommend”, “this”, “book”].  At the same time it needs to skip the next iteration so the next word does not appear two times.
> 2. It needs to be recursive: we might encounter multiple ngram words in a row. Then all of the words needs to be concatenated into a single ngram. So [“This”,”is”,”a”,”very”,”very”, “good”,”book”] needs to be concatenated in [“This”,”is”,”a”,”very very good”, “book”]. If n words are concatenated together into a single n-gram, the next n iterations need to be skipped.
> 3. In addition to concatenating words with the words following it, it might also be interesting if we concatenating it with the word preceding it. For example, forming n-grams including the word “book” and its preceding words leads to features like “worst book”, “best book”, “fascinating book” etc…

e.g. a function which generates n-grams from the splitted text and the list of specified ngram words:

```python
def generate_ngrams(text, ngram_words):
    new_text = []
    index = 0
    while index < len(text):
        [new_word, new_index] = concatenate_words(index, text, ngram_words)
        new_text.append(new_word)
        index = new_index+1 if index!= new_index else index+1
    return new_text

def concatenate_words(index, text, ngram_words):
    word = text[index]
    if index == len(text)-1:
        return word, index
    if word in bigram_array:
        [word_new, new_index] = concatenate_words(index+1, text, ngram_words)
        word = word + ' ' + word_new
        index = new_index
    return word, index
```

Here concatenate_words is a recursive function which either returns the word at the index position in the array, or the word concatenated with the next word. It also return the index so we know how many iterations need to be skipped.

This function will also work if we want to append words to its previous words. Then we simply need to pass the reversed text to it text = list(reversed(text)) and concatenate it in reversed order: word = word_new + ' ' + word.

We can put this information together in a single function, which can either concatenate with the next word or with the previous word, depending on the value of the parameter ‘forward’:
```python
def generate_ngrams(text, ngram_words, forward = True):
	new_text = []
	index = 0
	if not forward:
		text = list(reversed(text))
	while index < len(text):
		[new_word, new_index] = concatenate_words(index, text, ngram_words, forward)
		new_text.append(new_word)
		index = new_index+1 if index!= new_index else index+1
	if not forward:
		return list(reversed(new_text))
	return new_text

def concatenate_words(index, text, ngram_words, forward):
	words = text[index]
	if index == len(text)-1:
		return words, index
	if words.split(' ')[0] in bigram_array:
		[new_word, new_index] = concatenate_words(index+1, text, ngram_words, forward)
		if forward:
			words = words + ' ' + new_word
		else:
			words = new_word + ' ' + words
		index = index_new
	return words, index
```

