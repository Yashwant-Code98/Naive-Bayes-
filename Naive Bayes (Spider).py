# importing a required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# load a dataset
heart = pd.read_csv('heart.csv')
# view first 5 records of dataset
print(heart)
# view number of rows and columns in dataset
print(heart.shape)
# view percentiles of columns
print(heart.describe())
# view basic informartion about dataset
print(heart.info())
# view minimum value in column
print(heart['chol'].min())
# view maximum value in column
print(heart['chol'].max())
# view unique value in column
print(heart['trestbps'].unique())
# check duplicate value in column
print(heart.cp.duplicated())
# check the count of duplicate value
print(heart.cp.duplicated().sum())
# check duplicate values for entire dataset
print(heart.duplicated())
# check the count of duplicate value
print(heart.duplicated().sum()) 
# check non duplicate values
print(~heart.duplicated().sum())
# check missing values in dataset
print(heart.isnull().sum())
# check the count of only numeric column
print(heart.select_dtypes(include=['int64','float64']).columns) 
# check the count of only categorical column
print(heart.select_dtypes(include=['object']).columns)
# check values indexing range in columns
print(heart.select_dtypes(include=['int64','float64']).index)
# EDA On Heart Dataset
print(heart['age'].value_counts())
# visualize the age column
sns.histplot(heart['age'], color='red', linewidth=5,)
plt.title('Analysis On Age')
sns.set()
plt.show() 
print(heart['sex'].value_counts())
sns.countplot(heart['sex'])
plt.title('Analysis On Sex')
sns.set()
plt.show()
print(heart['cp'].value_counts())
plt.bar(list((0,2,1,3)),list(heart['cp'].value_counts()),color=['green','purple','cyan','orange'])
plt.title('Analysis On Chest Pain')
plt.show() 
print(heart['trestbps'].value_counts())
sns.kdeplot(heart['trestbps'], color='yellow', fill='yellow')
plt.title('Analysis On Trest BPS')
plt.show()
print(heart['chol'].value_counts())
sns.histplot(heart['chol'], color='k')
plt.title('Analysis On Cholistrol')
plt.show()
print(heart['target'].value_counts())
plt.bar(list((0,1)),list(heart['target'].value_counts()),color='brown')
plt.title('Analysis On Target')
plt.show()
# Age Vs Chest Pain
sns.scatterplot(x='age', y='cp', data=heart)
plt.title('Age Vs Chest Pain')
plt.show() 
# Naive Bayes 
# split the model into train and test
x = heart[['age']]
y = heart[['target']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
# view training dataset
print(x_train.head())
print(y_train.head())
# view testing dataset
print(x_test.head())
print(y_test.head())
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
print(mnb.fit(x_train,y_train))
y_pred = mnb.predict(x_test)
# view actual values 
print(y_test.head())
# view predicted values
print(y_pred[0:5])
#finding the residual by confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
# calculate the model accuracy
print((0+46)/(0+46+45+0))
