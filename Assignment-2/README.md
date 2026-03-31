# Iris Classification using K Nearest Neighbors



## dataset
The Iris dataset is a well known dataset available in scikit learn. It contains 150 samples with 4 features each:

 Sepal Length
 Sepal Width
 Petal Length
 Petal Width

There are 3 classes:

 Setosa
 Versicolor
 Virginica



### data Loading

The dataset is loaded using the load_iris function from sklearn.datasets.

### Train Test Split

The dataset is split into training and testing sets using train_test_split.
80 percent of the data is used for training and 20 percent for testing.

### Feature Scaling

StandardScaler is used to normalize the feature values.
This step is important because KNN is distance based and performs better when features are on the same scale.

### Model Selection

Different values of k from 1 to 9 are tested using cross validation.
The value of k with the highest average accuracy is selected.

### Model Training

The KNeighborsClassifier is trained using the optimal k value on the training data.

### Prediction

The trained model predicts the classes for the test dataset.

### Evaluation

The model performance is evaluated using:

 Accuracy score
 Confusion matrix
 Classification report

### Visualization

A grid visualization is created to display:

 Feature values of each sample
 Actual class label
 Predicted class label

Each sample is color coded:

 Green indicates correct prediction
 Red indicates incorrect prediction

## Requirements

The following libraries are required:

 numpy
 matplotlib
 scikit learn






## Conclusion

This assignment demonstrates the complete workflow of a machine learning classification problem using K Nearest Neighbors, including preprocessing, model selection, training, evaluation, and visualization.
