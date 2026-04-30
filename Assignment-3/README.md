#  Spam Detection using Naive Bayes

##  Overview

This project builds a machine learning model to classify messages as "Spam" or "Not Spam (Ham)" using the Naive Bayes algorithm.
It uses text processing techniques to convert messages into numerical form and then applies classification.


## Objective

The goal of this project is to:

 Automatically detect spam messages
 Reduce unwanted or harmful content
 Demonstrate text classification using machine learning

## Dataset

Dataset used: SMS Spam Collection Dataset
Source: UCI Machine Learning Repository
Total messages: ~5,574
Labels:

   spam → unwanted messages
   ham → normal messages

## Technologies Used

 Python
 Google Colab / Jupyter Notebook
 Libraries:

   pandas
   scikit-learn
   requests
   zipfile

##  Project Workflow

1.  Data Collection

    Downloaded dataset from UCI repository

2. Data Preprocessing

    Load dataset into DataFrame
    Convert labels (ham = 0, spam = 1)

3. Feature Extraction

    Convert text into numerical data using CountVectorizer

4.Train-Test Split

    80% training data
    20% testing data

5.Model Training

   Using  Multinomial Naive Bayes


7.Prediction

   Classify new messages as spam or not spam

## Model Used

"Multinomial Naive Bayes"

Works well for text classification
Based on probability


## Results

Accuracy: 96% – 99%

## Example Predictions

Message                   Result   

"You won a free prize!"   Spam     
"Let's meet tomorrow"     Not Spam 



## Limitations

* Cannot fully understand context
* May misclassify tricky messages
* Dataset is SMS-based (not full emails)



## Conclusion

This project demonstrates that a simple machine learning model like Naive Bayes can effectively classify spam messages with high accuracy.


