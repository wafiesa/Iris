# Spark Machine Learning For Iris Dataset

## Project Overview 

This project proposes to perform Machine Learning of Iris dataset using Spark MLlib. The Iris dataset is a classic in the field of machine learning, containing 150 samples divided among three different species of the iris flower: Setosa, Versicolor and Virginica. 

Spark MLlib facilitates the implementation of this classifier in a distributed computing environment, allowing for scalable and efficient handling of data. 

It includes features for handling data splitting, feature extraction and model evaluation, thus, making it suitable for large datasets and real time processing. 

![IrisFlower.jpeg](https://drive.google.com/uc?export=view&id=1-GAGFQCCjsReTfTfSkEDtZmCrG_wLjFK)

**Image 1: Iris Flower**
**photo credit to owner**

## Random Forest Classifier

Random Forest Classifier has been chosen to deliver predictioning outcomes and classifying information of the Iris dataset.  

This selection is based on the Iris dataset that has a discrete labels or in this context refers to different species of Iris flowers.  

## Code and Resources Used

* Hortonworks HDP Sandbox Version: 2.6.5.0
* Putty Version: 0.81
* Spark2 Version: 2.3.0
* Google Colab 
* Packages: pandas, numpy, matplotlib, seaborn, sklearn, pyspark

## Dataset Information

* [_**'iris.csv'**_](https://drive.google.com/file/d/1-DvyGJlpQYD4kNXK3sRh62pUu8dla0xo/view?usp=drive_link) contains iris dataset generated from RStudio.

#### Load The Dataset

The dataset can be uploaded by putting together the command below:

#### Command Prompt

```
Microsoft Windows [Version 10.0.22631.3593]
(c) Microsoft Corporation. All rights reserved.

C:\Users\Name>cd C:\ProgramData\Microsoft\Windows\Start Menu\Programs\PuTTY (64-bit)

C:\ProgramData\Microsoft\Windows\Start Menu\Programs\PuTTY (64-bit)>pscp -P 2222 "C:\Users\Name\Desktop\Data Management\iris.csv" maria_dev@127.0.0.1:iris.csv
```

#### PuTTY Commands

```
hadoop fs -copyFromLocal iris.csv /Name/iris.csv
```

#### Install Modules

```
sudo python2 -m install -U pandas
sudo python2 -m install -U numpy
sudo python2 -m install -U scikit-learn
```

#### Load Iris Dataset into Spark DataFrame
In PuTTY environment, create a script (vi Iris-RFC.py) to depict python code in Spark environment as below: 

```
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorIndexer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

spark = SparkSession.builder.getOrCreate()

# Load the iris dataset
iris = spark.read.csv("/user/maria_dev/wafiuddin/iris.csv", inferSchema=True, header=True)

# Print Top 10 rows
print("Top 10 rows iris:", iris.show(10))
```
Output:
Top 10 rows of iris:

|------------|-----------|------------|-----------|-------|
|Sepal.Length|Sepal.Width|Petal.Length|Petal.Width|Species|
|------------|-----------|------------|-----------|-------|
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
|         4.7|        3.2|         1.3|        0.2| setosa|
|         4.6|        3.1|         1.5|        0.2| setosa|
|         5.0|        3.6|         1.4|        0.2| setosa|
|         5.4|        3.9|         1.7|        0.4| setosa|
|         4.6|        3.4|         1.4|        0.3| setosa|
|         5.0|        3.4|         1.5|        0.2| setosa|
|         4.4|        2.9|         1.4|        0.2| setosa|
|         4.9|        3.1|         1.5|        0.1| setosa|
|------------|-----------|------------|-----------|-------|

```
# Convert the Spark DataFrame to a pandas DataFrame
iris_df = iris.toPandas()

# Show To 10 rows
print("Top 10 rows pandas DataFrame of iris_df:", iris_df.head(10))
```

#### Split The Dataset Into Training and Testing Sets
```
# Convert the Species column to a numerical format 
iris_df['Species'] = pd.factorize(iris_df['Species'])[0]
X = iris_df.drop(['Species'], axis=1)
y = iris_df['Species']

# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
```

#### Select Random Forest Classifier
```
# Create a Random Forest Classifier
RFC = RandomForestClassifier()
```

#### Cross Validation and Grid Search
```
# Define the hyperparameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(RFC, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding accuracy
print("Best Hyperparameters:", grid_search.best_params_)
``` 
Output:
('Best Hyperparameters:', {'min_samples_split': 2, 'n_estimators': 10, 'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 4})

#### Evaluations Metrics (Accuracy, Precision, Recall, F1-Score) 
```
# Make predictions with the best estimator
best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)

# Evaluate the performance of the model using various metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print("Accuracy of the model:", accuracy)
print("Precision of the model:", precision)
print("Recall of the model:", recall)
print("F1-score of the model:", f1_score)
```
Output
+ ('Accuracy of the model:', 0.9666666666666667)
+ ('Precision of the model:', 0.9714285714285714)
+ ('Recall of the model:', 0.9666666666666667)
+ ('F1-score of the model:', 0.9672820512820512)

🔶 Insights: The outputs represent four key evaluation metrics for a classification model: accuracy, precision, recall and F1-score. Accuracy, at 96.67%, indicates that the model correctly predicted the majority of the test samples. Precision, slightly higher at 97.14%, reflects that when the model predicts a class, it is correct most of the time. Recall, identical to accuracy, signifies that the model successfully identifies most of the relevant cases across all classes. The F1-score, at 96.73%, balances precision and recall, suggesting a harmonious blend of both in terms of prediction reliability and class coverage. These metrics collectively demonstrate that the model performs exceptionally well across various aspects of classification accuracy.

#### Precision model for best prediction since the highest percentage 97.14%
```
# Use Precision model to generate prediction since the highest percentage 97.14%
grid_search = GridSearchCV(RFC, param_grid, cv=5, scoring='precision_weighted')
grid_search.fit(X_train, y_train)

# Regenerate the best hyperparameters and the corresponding to precision
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Precision Score:", grid_search.best_score_)

# Make predictions with the best estimator
best_estimator_precision = grid_search.best_estimator_
y_pred = best_estimator_precision.predict(X_test)

# Evaluate the performance of the Precision model
precision = metrics.precision_score(y_test, y_pred, average='weighted')
print("Tuned Precision Score:", precision)
```
Output
+ ('Best Hyperparameters:', {'min_samples_split': 2, 'n_estimators': 50, 'bootstrap': True, 'max_depth': 5, 'min_samples_leaf': 2})
+ ('Best Precision Score:', 0.9777083333333333)
+ ('Tuned Precision Score:', 0.9714285714285714)

🔶 Insights: The outputs represent an optimised precision RandomForestClassifier via GridSearchCV, targeting the highest weighted precision. The best hyperparameters achieved during the training are shown, including settings for sample splitting, estimator count, bootstrap usage and depth parameters. The best cross-validated precision score from the grid search is approximately 97.77%. The tuned model, tested on unseen data, yields a precision score of about 97.14%. This indicates that the model, when predicting, is highly accurate in terms of classifying instances correctly across different classes.

#### F1-Score model for best prediction since second highest percentage 96.72%
```
# Use F1-Score model to generate prediction since second highest percentage 96.72%
grid_search = GridSearchCV(RFC, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Regenerate the best hyperparameters and the corresponding to F1-Score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best F1-Score:", grid_search.best_score_)

# Make predictions with the best estimator
best_estimator_f1 = grid_search.best_estimator_
y_pred = best_estimator_f1.predict(X_test)

# Evaluate the performance of the Precision model
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
print("Tuned F1-Score:", f1_score)
```
Output
+ ('Best Hyperparameters:', {'min_samples_split': 10, 'n_estimators': 10, 'bootstrap': True, 'max_depth': 5, 'min_samples_leaf': 4})
+ ('Best F1-Score:', 0.966506972559604)
+ ('Tuned F1-Score:', 0.9672820512820512)

🔶 Insights: The provided code snippet employed GridSearchCV to optimise a RandomForestClassifier based on the weighted F1-Score, selecting parameters that best balance precision and recall. The grid search yields optimal hyperparameters focused on tree depth, sample splits, and the number of estimators. The best F1-Score from the training phase is approximately 96.65%. When applied to test data, the best estimator achieves a slightly higher F1-Score of about 96.73%, indicating effective generalisation and a robust ability to handle class imbalance and maintain accuracy across labels.

#### Comparison Between Predicted Labels and Actual Labels

Performing confusion matrix can help us to understand the comparison between Predicted Labels and Actual Lables.

Thus, we can write the code to generate best visualisation as below:

```
# Print the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Transform to df for easier plotting
cm_df = pd.DataFrame(conf_matrix,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
```
 
![Heatmap](https://drive.google.com/uc?export=view&id=1KxwqTQxK4K8wm5UiS62EEtRQIFVIIP-e)

**Image 2: Confusion Matrix**

🔶 Insights: The confusion matrix shows the classification results for three classes: setosa, versicolor and virginica. While setosa and versicolor are perfectly classified (11 and 12 correct predictions), virginica has 6 correct predictions but one instance is misclassified as versicolor. This indicates excellent model performance for setosa and versicolo meanwhile good for virginica.

## Recommendations

For classifying the Iris dataset, several alternative machine learning approaches can be explored beyond RandomForestClassifier. For example, K-Nearest Neighbors (KNN) is simple yet effective technique that classifies based on a similarity measure such as distance functions.

## Conclusion

The selection of the RandomForestClassifier on the Iris dataset has demonstrated excellent effectiveness, highlighted by outputs across multiple evaluation metrics. The model achieved high scores in precision, F1-score, accuracy and recall. This outcome illustrates, its strong ability to correctly classify the three Iris species which are setosa, versicolor and virginica. 

This success is attributed to the machine learning method of the RandomForest which combines multiple decision trees to reduce overfitting and enhance prediction accuracy. The optimal set of hyperparameters was identified through a comprehensive GridSearchCV process, ensuring the model's reliability and adaptability to the dataset.

These characteristics make the RandomForest an ideal choice for the Iris dataset, providing a model that is not only accurate but also robust against the variability within the data. The outcome confirms that RandomForest is a powerful tool for multiclass classification, capable of delivering insightful and dependable results in botanical classification tasks.
