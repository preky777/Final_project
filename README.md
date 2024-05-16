# Data Science Portfolio

This repository contains four distinct data science projects, each tackling a unique problem using various data science techniques and methodologies.

# Project 1: Twitter Sentiment Analysis

## Problem Statement
Analyze the sentiments of tweets on Twitter. The dataset contains 1,600,000 tweets extracted using the Twitter API, annotated with sentiment labels (0 = negative, 2 = neutral, 4 = positive).

## Dataset

    •Fields: target, ids, date, flag, user, text
    
## Objective

Design a classification model to correctly predict the polarity of the tweets.

## Explanation

The process of cleaning and creating the classification model is documented in twitter_sa.ipynb.

## Model Saving

    import pickle
    
    # Save the trained classifier
    with open('logistic_regression_model.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)
    
    # Save the vectorizer
    with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectoriser, vectorizer_file)
    
    # Save the preprocessing function
    with open('preprocess_tweet_function.pkl', 'wb') as preprocess_file:
        pickle.dump(preprocess_tweet, preprocess_file)


# Project 2: Ratings of Guvi Courses

## Problem Statement
Analyze Guvi course data to predict the ratings given by learners.

## Dataset
    •Fields: course_title, url, price, num_subscribers, num_reviews, num_lectures, level, rating, content_duration, published_timestamp, subject

## Objective
Design a regression model to predict course ratings.

## Explanation
The regression model is built in ratings.ipynb.

## Model Saving

    import pickle
    
    # Save the model
    with open('random_forest_regressor_model.pkl', 'wb') as file:
        pickle.dump(rf_regressor, file)
    
    # Save the StandardScaler
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    
    # Save the OneHotEncoder for 'level'
    with open('ohe_level.pkl', 'wb') as file:
        pickle.dump(ohe, file)
    
    # Save the OneHotEncoder for 'subject'
    with open('ohe_subject.pkl', 'wb') as file:
        pickle.dump(ohe2, file)


# Project 3: Instagram Influencers

## Problem Statement
Analyze Instagram influencer data to answer various questions and visualize trends.

## Dataset
    •Fields: rank, channel_info, influence_score, posts, followers, avg_likes, 60dayeng_rate, newpostavg_like, total_likes, country
    
## Objectives

    1.Identify correlated features.
    2.Frequency distribution of influence score, followers, and posts.
    3.Determine the country with the highest number of influencers.
    4.Identify the top 10 influencers based on followers, average likes, and total likes.
    5.Describe the relationship between specific pairs of features.


## Explanation
All the answers are documented in a word document named instagram_infs.

# Project 4: Image Classification

## Problem Statement
Perform binary classification of images (cats vs. dogs).

## Dataset
Download the dataset from Kaggle.

## Objective
Design a binary classifier and evaluate it using appropriate metrics.

## Explanation
The classifier and its evaluation are documented in dog_cat_class_model.ipynb.


## Repository Structure

    .
    ├── project1_twitter_sentiment_analysis
    │   └── twitter_sa.ipynb
    │   └── logistic_regression_model.pkl
    │   └── tfidf_vectorizer.pkl
    │   └── preprocess_tweet_function.pkl
    ├── project2_guvi_course_ratings
    │   └── ratings.ipynb
    │   └── random_forest_regressor_model.pkl
    │   └── scaler.pkl
    │   └── ohe_level.pkl
    │   └── ohe_subject.pkl
    ├── project3_instagram_influencers
    │   └── instagram_infs.docx
    ├── project4_image_classification
    │   └── dog_cat_class_model.ipynb
    ├── README.md
    └── requirements.txt


## Installation
To run these projects locally, follow these steps:

    1.Clone the repository:
    git clone https://github.com/your_username/data-science-portfolio.git
    
    2.Navigate to the project directory:
    cd data-science-portfolio
    

## Usage

    1.Open the relevant Jupyter notebook file (.ipynb) for the project you wish to explore.
    2.Follow the instructions within the notebook to execute the code cells.
