# Movie-Reviews-Sentiment-Analysis-Rotten-Tomatoes-Dataset
Sentiment analysis on the Rottoen Tomatoes movie reviews dataset.

The dataset is taken from an old Kaggle competition: https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
Dataset size: 156060.

As of now my submission made to the competition is giving score of 0.62980 which stands at 245/862 (top 28%). I am expecting the score to improve on doing further feature engineering. I'll also be running the next model with Convolutional neural networks (CNNs).

Following table summarises the F1-Scores of the model runs:




                             | tf-idf | BoW (n=2) |

    MultiNomialNB            |  0.48  |   0.51    |
    LinearSVC                |  0.55  |   0.53    |
    LogisticRegression       |  0.55  |   0.54    |
    DecisionTreeClassifier   |  0.43  |   0.42    |
    
    (BoW = Bag of words)
