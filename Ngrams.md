# N-grams

## Links
1. [Sentiment Analysis with bag-of-words](http://ataspinar.com/2016/01/21/sentiment-analysis-with-bag-of-words/)
2. [Sentiment Analysis with bag-of-words (part 2)](https://ataspinar.wordpress.com/2016/02/01/sentiment-analysis-with-bag-of-words-part-2/)

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
