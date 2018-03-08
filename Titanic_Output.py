import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import re
import math
import seaborn as sns

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,LabelBinarizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from keras.models import Sequential
from keras.layers import Dense
import missingno as mn
from sklearn.model_selection import train_test_split
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
from numpy.random import seed

#import dataset
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()

test.head()

#Move survived column in the train dataset
cols = list(train.columns.values)
cols.pop(cols.index('Survived'))
train = train[cols+['Survived']]

#Combine train and test data
combined_dataset = train.append(test,ignore_index=True)

#Check for Fare distribution
sns.distplot(combined_dataset["Fare"].dropna())

#Check for Age distribution
sns.distplot(combined_dataset["Age"].dropna())

#Visualize Missing Data
mn.matrix(combined_dataset)

#Getting categorical column numbers and continuous column numbers
cols = combined_dataset.columns
num_cols = combined_dataset._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))

#Segregating categorical and conitnuous
df_numeric = combined_dataset[num_cols]
df_cat = combined_dataset[cat_cols]

#Separating survived data into another variable
y = df_numeric['Survived']
df_numeric = df_numeric.drop(['Survived'], axis=1)

#Imputing missing age
imputed_df = pd.DataFrame(KNN(3).complete(df_numeric))

#Column name fix
imputed_df.columns = df_numeric.columns
imputed_df.index = df_numeric.index

#Check for Age distribution after imputation
sns.distplot(imputed_df["Age"].dropna())

#Splitting name into Last Name and Title
df_cat['LastName'] = df_cat['Name'].str.extract(r"([A-Za-z0-9 _]+)",expand=False)
df_cat['Title'] = df_cat.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#Converting titles 
df_cat['Title'] = df_cat['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_cat['Title'] = df_cat['Title'].replace('Mlle','Miss')
df_cat['Title'] = df_cat['Title'].replace('Ms','Miss')
df_cat['Title'] = df_cat['Title'].replace('Mme','Mrs')

#Analyzing and combining family structures
imputed_df['FamilySize'] = imputed_df['Parch'] + imputed_df['SibSp']

#Check for dataframe datatype
imputed_df['Pclass'] = imputed_df.Pclass.astype('int')
#imputed_df['Pclass'] = imputed_df.Pclass.astype('str')
imputed_df.dtypes

#Imputing Cabin data for missing values
df_cat['Cabin'][df_cat.Cabin.isnull()] = 'U0'
df_cat['Cabin_No'] = df_cat.Cabin.str[:1]

#Display rows with missing values for categorical variables
df_cat[df_cat.isnull().any(axis=1)]

#Replace missing categorical embarked data
df_cat = df_cat.fillna({'Embarked':'S'})

#Segregating categorical data
df_cat = pd.concat([df_cat,imputed_df['Pclass']],axis=1)
imputed_df = imputed_df.drop(['Pclass'], axis=1)

#Dropping Ticket, Cabin, Name and LastName
df_cat.drop(['Ticket','Name','Cabin','LastName'], axis=1, inplace=True)

#Saving PassengerID in a list to be used to create the final output
PassengerId = imputed_df['PassengerId']
imputed_df = imputed_df.drop(['PassengerId'],axis=1)

#Creating dummy variables
df_cat = pd.get_dummies(df_cat,drop_first = True)

#Concatenating imputed numeric dataframe with the categorical dataframe
df = pd.concat([imputed_df,df_cat,y],axis=1)

#Dropping irrelevant fields
df.drop(['Parch','SibSp'], axis=1, inplace=True)

df.describe()

#To transform to normal distribution
df["Fare"] = np.log1p(df["Fare"])

#Visualizing FamilySize with Survival Rates
sns.set(style="whitegrid")
sns.factorplot('Survived',data = df, hue = 'FamilySize', kind = 'count')

#Visualizing FamilySize vs Survived vs PClass
sns.barplot(x = 'FamilySize', y = 'Survived',data = df, hue = 'Pclass')

sns.set_style("whitegrid")
sns.barplot(x="Pclass",y="Survived",data=train)

sns.set_style("whitegrid")
sns.barplot(x="Embarked",y="Survived",data=train)

pd.crosstab(train['Embarked'],train['Pclass'])

#Creating train and test
X_train = df[df['Survived'].notnull()]
X_test = df[df['Survived'].isnull()]

#Separating survived data into another variable
y_train = X_train['Survived']
y_train = y_train.astype('int')
y_train = y_train.astype('str')
X_train = X_train.drop(['Survived'], axis=1)
y_test = X_test['Survived']
X_test = X_test.drop(['Survived'], axis=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create a dictionary containing all the candidate values of the parameters
parameter_grid = dict(n_estimators=list(range(100, 200, 300)),
                      criterion=['gini','entropy'],
                      max_features=[0.85,1.0, 1.5],
                      max_depth= [None] + list(range(10, 15, 25)),
                      min_samples_leaf=list(range(3,7)),
                      min_samples_split=list(range(3,7)))

# Creata a random forest object
random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)

# Create a gridsearch object with 10-fold cross validation, and uses all cores (n_jobs=-1)
clf = GridSearchCV(estimator=random_forest, param_grid=parameter_grid, cv=10, verbose=1, n_jobs=-1)

# Retrain the model on the whole dataset
clf.fit(X_train, y_train)

# View the accuracy score
print('Best score for X_train:', clf.best_score_)
# View the best parameters for the model found using grid search
print('Best Estimator:',clf.best_estimator_.n_estimators) 
print('Best Max Features:',clf.best_estimator_.max_features)
print('Best Max Depth:',clf.best_estimator_.max_depth)
print('Best Min Sample Leaf:',clf.best_estimator_.min_samples_leaf)
print('Best Min Sample Split:',clf.best_estimator_.min_samples_split)

# Set the random seed
random_seed = 2

#Finding the best parameters
n_estimators_rf = clf.best_estimator_.n_estimators
max_features_rf = clf.best_estimator_.max_features
max_depth_rf = clf.best_estimator_.max_depth
min_samples_leaf_rf = clf.best_estimator_.min_samples_leaf
min_samples_split_rf = clf.best_estimator_.min_samples_split

#Using the randomforestclassifier
random_forest_pipeline = RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=max_features_rf, min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)

# Predict who survived in the test dataset
rf = random_forest_pipeline.fit(X_train, y_train)
clf.score(val_x, val_y)

#Predicting the classes
results = random_forest_pipeline.predict(X_test)

#Creating the data frame for submission
PassengerId = test['PassengerId']
PassengerId = PassengerId.astype('str')
final_output = pd.DataFrame({'PassengerId':PassengerId.tolist(),'Survived':results.tolist()})

#Final output file
final_output.to_csv('final_output.csv', index = False)