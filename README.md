# Multiclass-Twitter-Sentiment-Analysis

This project leverages a sequential Keras model to categorize tweets related to COVID-19 into five distinct sentiment classes. The model employs GloVe word embeddings and a Bidirectional LSTM architecture for accurate sentiment analysis. The goals is to develop a strong sentiment analysis model to precisely categorize the sentiment of diverse tweets.

## Dataset

The model was trained on a dataset from the ["Pandemic Tweet Challenge"](https://www.kaggle.com/competitions/pandemic-tweet-challenge/overview), which classifies tweets into five sentiment categories: Extremely Negative, Negative, Neutral, Positive, and Extremely Positive. This dataset provides valuable insights into public sentiment during the pandemic.

## Dataset Format (Data Example):


| Serial        | UserName         | ScreenName       | Location         | TweetAt          | OriginalTweet                                           | Sentiment    	   |
| ------------- |:----------------:|:----------------:|:----------------:|:----------------:|:-------------------------------------------------------:|:----------------:|	
| 0             |3799	             |48751             |London            |13-03-2020	      |@MeNyrbie @Phil_Gahan @Chrisitv https://t.co/i...        |Neutral           |
| 1             |3800	             |48752             |UK                |12/3/2020	        |advice Talk to your neighbours family to excha...        |Positive          |
| 2             |3801	             |48753             |Vagabonds         |13-03-2020	      |Coronavirus Australia: Woolworths to give elde...        |Positive          |
| 3             |3802	             |48754             |NaN               |14-03-2020	      |My food stock is not the only one which is emp...  	    |Positive          |
| 4	            |3803	             |48755             |NaN  	           |13-03-2020	      |Me, ready to go at supermarket during the #COV...	      |Extremely Negative|


### The dataset is stored in a CSV format with two important columns:
- OriginalTweet: The original text of the tweet. (Feature Column)
- Sentiment: The sentiment class label associated with each tweet. (Target Column)
- Other columns are irrelevant, hence are dropped.

## Preprocessing

Tweets are cleaned to eliminate irrelevant information, reduce noise and nuances. Such as,
- Text is lowercased.
- Contractions are expanded.
- URLs, mentions, and special characters are removed.

Other pre-processing tasks include,

- Tokenization is applied to convert tweets into sequences of words.
- Padding ensures all sequences are of equal length.
- Word Embeddings: GloVe embeddings ([glove.6B.100d](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt)) are used to map words to vectors.
 

