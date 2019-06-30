# Sentiment Analysis

## Links

1. [VADER Sentiment Analysis Explained](http://datameetsmedia.com/vader-sentiment-analysis-explained/)
2. [The Predicting Power of Textual Information on Financial Markets](https://www.comp.hkbu.edu.hk/~cib/2005/Jun/iib_vol5no1_article1.pdf)
3. [CS224N Final Project: Sentiment analysis of news articles for financial signal prediction](https://nlp.stanford.edu/courses/cs224n/2011/reports/nccohen-aatreya-jameszjj.pdf)
4. [Sentiment Analysis of Financial News Articles using Performance Indicators](https://arxiv.org/pdf/1811.11008.pdf)
5. [StockSentimentTrading](https://github.com/jasonyip184/StockSentimentTrading/blob/master/.ipynb_checkpoints/Stock%20Sentiment%20Algo%20Trading-checkpoint.ipynb)
6. [ELMo for Text Classification in Python](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/)
7. [Sentiment Analysis with bag-of-words](http://ataspinar.com/2016/01/21/sentiment-analysis-with-bag-of-words/)
##  A bag-of-words model for text classification

n-gram

## VADER Sentiment Analysis

[github](https://github.com/cjhutto/vaderSentiment)

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. 

> VADER (Valence Aware Dictionary for sEntiment Reasoning) is a pre-built sentiment analysis model included in the NLTK package. It can give both positive/negative (polarity) as well as the strength of the emotion (intensity) of a text. It is rule-based and relies heavily on humans rating texts via Amazon Mechanical Turk — a crowd-sourcing e-platform which utilizes human intelligence to perform tasks that computers are currently unable to do. This literally means that other people have already done the dirty work of building a sentiment lexicon for us. These are words or any textual form of communication generally labelled according to their semantic orientation as either positive or negative) for us.

The human raters of Vader used 5 heuristics to analyze the sentiment:
1. Punctuation — I love pizza vs I love pizza!!
2. Capitalization — I’m hungry!! vs I’M HUNGRY!!
3. Degree modifiers (use of intensifiers)— I WANT TO EAT!! VS I REALLY WANT TO EAT!!
4. Conjunctions (shift in sentiment polarity, with later dictating polarity) — I love pizza, but I really hate Pizza Hut (bad review)
5. Preceding Tri-gram (identifying reverse polarity by examining the tri-gram before the lexical feature— Canadian Pizza is not really all that great.


### Loughran and McDonald Sentiment Word Lists

e.g. get the sentiment from a passage.
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
polarity_scores(passage)['compound']
```
