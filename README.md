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

A diagram of developed the model is given below

![Developed TF model](https://raw.githubusercontent.com/RezuwanHassan262/Multiclass-Twitter-Sentiment-Analysis/figs/model_arch.png) 

(https://raw.githubusercontent.com/RezuwanHassan262/Multiclass-Twitter-Sentiment-Analysis/refs/heads/main/figs/model_arch.png?token=GHSAT0AAAAAACZXN5HUNXGGZZ4ZHACPIDBOZZDO6CA) 

 
## Results

After a lot of trials and errors with the parameters and model architecture, this model was finalized.

### Model Performance

We can observe the model's performance on training and validation data, the learning process and it's generalization ability. 

![Model train-Loss Curves](https://raw.githubusercontent.com/RezuwanHassan262/Multiclass-Twitter-Sentiment-Analysis/refs/heads/main/figs/model_arch.png?token=GHSAT0AAAAAACZXN5HVW7ZIZARZP5WKDN7MZZIMKUQ) 


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
| Extremely Negative  | ----      | ----   | ----     | ----    |
| Extremely Positive  | ----      | ----   | ----     | ----    |
| Negative            | ----      | ----   | ----     | ----    |
| Neutral             | ----      | ----   | ----     | ----    |
| Positive            | ----      | ----   | ----     | ----    |

| Metric        | Value |
|---------------|-------|
| Accuracy      | ----  |
| Macro Avg     | ----  |
| Weighted Avg  | ----  |





## Improvements

### Future work scopes:

    - Implementing transformer-based models like BERT to enhance sentiment classification precision.
    - Employing data augmentation methods, including back-translation and synonym replacement, to expand tweet datasets.
    - Conducting hyperparameter optimization to enhance model performance.


## Explainable AI Integration (LIME: Local Interpretable Model-agnostic Explanations)

I aimed to enhance the interpretability of the model's predictions by integrating eXplainable AI (XAI) techniques. Specifically, I focused on implementing LIME (Local Interpretable Model-Agnostic Explanations). LIME works by approximating the complex model's behavior locally around a specific instance, creating a simpler, interpretable model to explain the prediction.

By incorporating LIME, I sought to understand the factors influencing the model's decisions. This would improve the model's transparency and help identify potential biases or errors.

