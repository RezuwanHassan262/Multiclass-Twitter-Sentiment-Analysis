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
 

## Model Architecture

The model is consisting of multiple layers including:

- Embedding Layer: Leveraged GloVe word embeddings to give meaning to words in the model.
    Bidirectional LSTM Layers: Employs a bidirectional approach to capture long-range dependencies in both forward and backward directions.
    Dropout and Batch Normalization: Regularization techniques to reduce model complexity and improve generalization.
    Fully Connected Layers: To integrate the extracted insights into a unified representation for decision-making.

%%%%%%%%%%%%%%%%%%%% image %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
## Results

After a lot of trials and errors with the parameters and model architecture, this model was finalized.

### Model Performance


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Curves %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


Model performance on test data.

    Training Accuracy: Approximately 85% - 86%
    Validation Accuracy: Approximately 79.35% - 81.60%
    Test Accuracy: Approximately 80.25% - 80.79%

### Evaluation Metrics

- Confusion Matrix: A confusion matrix is employed to assess the model's performance by visualizing the distribution of true positive, true negative, false positive, and false negative predictions for each sentiment category.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

- Classification Report: A classification report analyzes the model's precision, recall, and F1-score for each sentiment category.

## Model Performance Metrics

![Model Performance Metrics]

| Label               | Precision | Recall | F1-score | Support |
|---------------------|-----------|--------|----------|---------|
| Extremely Negative  | 0.77      | 0.87   | 0.82     | 548     |
| Extremely Positive  | 0.78      | 0.88   | 0.83     | 663     |
| Negative            | 0.77      | 0.76   | 0.76     | 991     |
| Neutral             | 0.91      | 0.85   | 0.88     | 772     |
| Positive            | 0.80      | 0.75   | 0.77     | 1142    |

| Metric        | Value |
|---------------|-------|
| Accuracy      | 0.81  |
| Macro Avg     | 0.81  |
| Weighted Avg  | 0.81  |





## Improvements

### Future work scopes:

    - Implementing transformer-based models like BERT to enhance sentiment classification precision.
    - Employing data augmentation methods, including back-translation and synonym replacement, to expand tweet datasets.
    - Conducting hyperparameter optimization to enhance model performance.


## Explainable AI (Local Interpretable Model-agnostic Explanations)


