
### Description

The objective of this data science wuery is to predict whether patient has diabetes or not. The diabetes dataset consists of several medical predictor (Independent) variable and one target variable (outcome). Predictor variables include 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction' and 'Age'.

Dataset URL: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

#### Name of columns with their meanings

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration a 2 hour in an oral glucose tolerance test
- BloodPressure: Diastolin blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function scores the probability of diabetes based on family history
- Age: Age(years)
Outcome: Class Variable( 0 or 1)

#### Step 1: Importing libraries like Numpy, Pandas, matplotlib, Seaborn and scikit learn(Sklearn)

Numpy: general purpose array processing package. It provides a high-performance multidimentional array object. It is the fundamental package for scientific computing with Python Numeric Analytics Library.

Pandas: used for dataset reading, data analytics and data processing. It's also used for data manipulation. 

Matplotlib: for charts or plots

Seaborn: for charts and plots built on top of Matplotlib. it is a data visualization library based on matplotlib. It provide high level interface for drawing attractive and informative statistical graphics.

Sklearn: scikit-learn provides many unsupervised and supervised learning algorithms like Regression, Classification, Model Selection Processing.

![import libraries](https://github.com/PoojaParab-DA/python/assets/172165136/c1a44105-3da8-4cd8-8ecc-5c4ce9c5461d)

#### Step 2: Loading datasets

![read csv](https://github.com/PoojaParab-DA/python/assets/172165136/5c7cc19c-6c0c-45b0-b38d-58852b85911a)


#### Step 3: Exploratory Data Analysis

Exploratory Data Analysis(EDA), also known as Data Exploration, is a step in the Data Analysis Process, where a number of techniques are used to better understand the dataset being used.

In this step, we will perform the below operations. 

##### 3.1 Understanding variables

3.1.1 Head of the dataset
3.1.2 The shape of the dataset
3.1.3 List types of all columns
3.1.4 Info of the dataset
3.1.5 Summary of the dataset

##### 3.2 Data cleaning

3.2.1 Check the duplicates
3.2.2 Check the NULL values

Code: 
To understand records of the dataset.

![head1](https://github.com/PoojaParab-DA/python/assets/172165136/8362b08b-5758-4af9-9630-28336d0a641d)
![tail](https://github.com/PoojaParab-DA/python/assets/172165136/89784355-f29f-42ac-91ee-f29a27ec46a8)
![sample](https://github.com/PoojaParab-DA/python/assets/172165136/fd286520-a6f1-475a-a8b6-3b342a393d9b)


- head() displays initial records of the dataset.
- tail() displays last records of the dataset.
- sample() displays any random record of the dataset.

To understand shape of the dataset:

![shape](https://github.com/PoojaParab-DA/python/assets/172165136/0fe86400-0d94-42b4-bbbb-c1fd47443e45)

shape function will give number of rows and columns. here, rows=768 and columns=9.

dtypes will give datatypes of each column.
![dtypes1](https://github.com/PoojaParab-DA/python/assets/172165136/7952aeac-5050-405f-b788-4ac11b9e37bf)

info() is used to check the information about the data and datatypes of each respective attribute. 

![info1](https://github.com/PoojaParab-DA/python/assets/172165136/f315b263-636b-42a6-a105-d007670111c3)

All are non-null values that means no missing values in the dataset.

#### Summary of the dataset

describe() function will give minimum value, mean value, maximum value and other percentages of each column.

![describe](https://github.com/PoojaParab-DA/python/assets/172165136/255269e7-9bb0-4da1-940f-10bb1cbd37cf)

We got the statistical summary of the dataset.

In the above stat summary we see that, min value of columns Glucose, BloodPressure, SkinThickness, Insulin, BMI is zero. As these values can't be zero, we will add mean values of these respective columns instead of zero.

In data cleaning, we will drop the duplicates.

![drop_duplicates](https://github.com/PoojaParab-DA/python/assets/172165136/75bd7c04-c467-4d32-91df-cc616a7caffe)
As the number of rows and col before and after the drop_duplicates function is same, there where no duplicates in the dataset.

In the next step of data cleaning, using isnull.sum() function we can see the null values present in the every column in the dataset.
![null values](https://github.com/PoojaParab-DA/python/assets/172165136/3be420df-0c4a-4c93-b70e-f9cfaa823bbd)
Here, we can see, no null values are found. 



Now, we will replace the zero values from Glucose, BloodPressure, SkinThickness, Insulin, BMI column.
![replace zero values](https://github.com/PoojaParab-DA/python/assets/172165136/2ca2212b-b9cc-4506-b1ad-74af2038fb13)

Checking bias: 
![checking bias](https://github.com/PoojaParab-DA/python/assets/172165136/f2709c81-0e45-423c-acab-e585356ba943)

- The above graph shows that the data is biased towards datapoints having outcome value as 0. 
- The number of non-diabetics is almost double the number of diabetic patients.
- Out of 768 people, 500 people are non-diabetic and 268 people are diabetic. 

#### Step 4: Analysing Relationship Between Variables

#### Histograms
We can check the distribution of the data using histograms- whether the data is normally distributed or it its skewed(to the left or right).

![hist1](https://github.com/PoojaParab-DA/python/assets/172165136/0ae9e19f-01ba-47d7-8a0e-22ce0248007c)
![hist2](https://github.com/PoojaParab-DA/python/assets/172165136/2e0dc285-ecd7-4a1b-ace5-f5d73e22be72)

Observations: 
- Glucose, BloodPressure, BMI are normally distributed.
- Insulin, Age, DiabetesPedigreeFunction are left skewed.

#### Scatter plots
Plot the values of two variables along two axis, like age and height. Scatterplots are useful for understanding relationships between two variables. 

![scatter1](https://github.com/PoojaParab-DA/python/assets/172165136/7444358e-6c31-429d-aad2-bb727d109cb1)
![scatter2](https://github.com/PoojaParab-DA/python/assets/172165136/901699e9-f993-4188-87a3-0fd297fd0c7c)

Observation: 
Few parameters are like BMI and SkinThickness are very well correlated while few are not correlated.

#### Pair plot
Pair Plot gives relationship between all of the variables.

It's easier to find out distribution between different outcomes as well as the relationship between different variables. 

![pairplot1](https://github.com/PoojaParab-DA/python/assets/172165136/91784084-4b93-467f-832c-6ca60a180a1c)
![pairplot2](https://github.com/PoojaParab-DA/python/assets/172165136/9ff8f6ad-c581-41bf-b1f2-df14b440a4ac)

#### Correlation Analysis: 

Correlation analysis is used to quantify the degree to which two variables are related. Through the correlation analysis, you evaluate correlation coefficient that tells you how much one variable changes when the other one does. Correlation analysis provides you with a linear relationship between two variables. When we correlate feature variables with the target variables, we get to know how much dependency is there between partucular feature variables and target variable.

![correlation](https://github.com/PoojaParab-DA/python/assets/172165136/a9db59c6-dbbd-44dd-8dfc-099b195c52e8)

From the correlation heatmap, we can see that there is a high correlation between Outcome and [Pragnancies,Glucose, BMI, Insulin, Age]. We can select these features to accept input from the user and predict the outcome. 


#### Step 5: Building prediction models

#### Split the data into x and y

Assign 'Outcome' column to y and rest other columns to x.
Check the initial entries using x.head() and y.head().
![split 1](https://github.com/user-attachments/assets/8d7e3fc3-ee8e-4ada-ae9c-b02629e7d7c5)
![split 2](https://github.com/user-attachments/assets/03dc5afb-9abb-44f1-aff8-ecf1bc06bae3)

Apply the StandardScaler from scikit-learn to standardize data X. The fit(X) method computes the mean and standard deviation for later use during the transformation. The transform(X) method applies the standardization transformation to data X. It subtracts the mean computed in the fit step and then scales by the standard deviation. The transformed data is stored in SSX.
![feature scalling](https://github.com/user-attachments/assets/6188f390-437b-45a3-995a-d083a57453b5)


The train_test_split function is used to split arrays or matrices (SSX) into random train and test subsets.

test_size=0.2: This parameter specifies the proportion of the dataset to include in the test split. Here, 0.2 means 20% of the data will be used for testing, and the remaining 80% will be used for training.

random_state=7: This parameter sets the seed for the random number generator used for shuffling the data before splitting. Providing a fixed random_state ensures reproducibility of the split.
![train test](https://github.com/user-attachments/assets/3f299165-2d96-4a77-82a2-5cdabb28c5f7)

#### Building classification model
Here, we will consider two models(Support Vector Classifier model and Random Forest Classifier model) and further select one model from these two. 
![classification model](https://github.com/user-attachments/assets/bd7740a8-f554-49bc-bc18-3a8e4eb912c7)

The fit(X_train, y_train) method of the SVC class and random forest fits the SVM model and rf model resp to the training data (X_train and y_train). This means it learns the patterns in X_train that correspond to y_train labels.

#### Making prediction
![prediction making](https://github.com/user-attachments/assets/3d3ffcfc-4fce-49c9-a7af-c03fd7efe707)
sv.predict(X_test) and rf.predict(X_test) are used to obtain predictions from our trained SVM and Random Forest models, respectively. These predictions are then compared and evaluated to determine the effectiveness of each model in making accurate predictions on unseen data. 

#### Model evaluation
![model evaluation](https://github.com/user-attachments/assets/bbb6710d-078c-4081-8dc1-ed0c88a31d01)

.score(X_train, y_train) calculates the accuracy of the SVM model (sv) on the training data (X_train and y_train).
.score(X_test, y_test) computes the accuracy of the SVM model (sv) on the test data (X_test and y_test)
accuracy_score(y_test, sv_pred) and accuracy_score(y_test, rf_pred) calculates the accuracy of the SVM and rf predictions compared to the true labels (y_test). 

We got 100% test accuracy in the random forest. But the overall accuracy score of random forest(75.97%) is less compared to that of SVM(83.11%). Hence, We will select SVM for further model building.


#### Confusion matrix
Its a table which is used to describe the performance of a classification problem. It visualizes the accuracy os a classifier by comparing predicted values with actual values. 
Following terms are used in confusion matrics:
- True Positive
- False Positive
- False Negative
- True Negative


![confusion matrix 1](https://github.com/user-attachments/assets/9455b87c-c847-43cf-942e-d4e65cc29913)
![confusion matrix 2](https://github.com/user-attachments/assets/0ca01f41-e881-4c6b-bf00-a944fa99db78)

sns.heatmap from the seaborn library helps visualize the confusion matrix with color intensity, making it easier to interpret the results.
Interpretation: In the heatmap:

- The diagonal elements (top-left to bottom-right) represent the number of correct predictions for each class.
- Off-diagonal elements indicate incorrect predictions.


Confusion Matrix Elements: These elements provide a detailed view of how well the SVM model is performing in terms of correctly and incorrectly classified instances.

Accuracy and Misclassification Rates: These metrics help quantify the overall performance of the model on the test set, providing insights into its predictive power.

- Accuracy Rate: Proportion of correctly classified instances (TN + TP) over the total number of instances.
- Misclassification Rate: Proportion of incorrectly classified instances (FP + FN) over the total number of instances.


#### Classification report 

![classification report](https://github.com/user-attachments/assets/ded291b1-4fc4-4c2e-932a-96130c540ef9)


The classification_report function from scikit-learn provides a detailed report showing various metrics for each class in classification problem.

Interpretation
- Precision: Indicates the proportion of positive predictions that were actually correct.
- Recall: Indicates the proportion of actual positives that were correctly predicted by the model.
- F1-score: Harmonic mean of precision and recall, providing a balanced measure between them.
- Support: Number of occurrences of each class in y_test.
- Accuracy: Overall accuracy of the model on the test set.

#### ROC AUC
![roc auc](https://github.com/user-attachments/assets/9d3ddb9a-c3be-46f7-9f8b-0b189392cf1e)
![roc auc chart](https://github.com/user-attachments/assets/f823da62-0a36-4f75-8b6e-8eb230d22302)

ROC AUC is a measure of the area under the ROC curve, which represents the model's ability to distinguish between positive and negative classes.

ROC Curve: This curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

AUC: The Area Under the Curve (AUC) quantifies the overall performance of the binary classification model.
