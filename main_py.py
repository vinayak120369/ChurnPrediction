# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 13:07:25 2022

@author: Vinayak Patil
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv('Churn_Modelling.csv')
print(df.info())
print(df.head())
print(df.dtypes)
print(df.shape)
dups = df.duplicated()
print(dups.any())
df.isnull().sum()
df.nunique()
#apply generic describe function
df.describe()
df_copy = df.copy()
df_copy.head()

df.drop(columns=['RowNumber','CustomerId','Surname'],axis = 0,inplace=True)
df_copy.drop(columns=['RowNumber','CustomerId','Surname'],axis = 0,inplace=True)
df_copy.head()

cat_columns = ['Geography','Gender']
for col in cat_columns:
    tempdf = pd.get_dummies(df_copy[col])
    df_copy = tempdf.merge(df_copy,left_index = True,right_index=True) 
    df_copy.drop(columns = col,inplace = True)
    
    
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,10)) 
sns.heatmap(df_copy.corr(),annot=True)
df['Exited'].value_counts()
plt.pie(df['Exited'].value_counts(),labels = ['Non-Exited','Exited'],autopct  = '%.2f %%')
plt.show()
df['Age'].hist()
plt.show()
temp_df = df.loc[df['Exited']==1,['Geography','Exited']]
temp_df = temp_df.groupby(by=['Geography'])['Exited'].count().reset_index()
temp_df1 = df.loc[df['Exited']==0,['Geography','Exited']]
temp_df1 = temp_df1.groupby(by=['Geography'])['Exited'].count().reset_index()
# plt.bar(temp_df['Geography'],temp_df['Exited'])
# plt.show()


fig, ax = plt.subplots(figsize=(4,8))
p1=ax.bar(temp_df['Geography'], temp_df['Exited'], 0.3,  label= 'Exited',color = 'salmon')
p2=ax.bar(temp_df1['Geography'], temp_df1['Exited'], 0.3, bottom=temp_df['Exited'],
       label='Not-Exited',color = 'lawngreen')
ax.set_ylabel('Exit_counts')
ax.bar_label(p1, label_type='edge',padding = -15)
ax.bar_label(p2, label_type='edge',padding = -15)

ax.legend()
plt.show()
temp_df = df.loc[df['Exited']==1,['Gender','Exited']]
temp_df = temp_df.groupby(by=['Gender'])['Exited'].count().reset_index()

temp_df1 = df.loc[df['Exited']==0,['Gender','Exited']]
temp_df1 = temp_df1.groupby(by=['Gender'])['Exited'].count().reset_index()

fig, ax = plt.subplots(figsize=(4,8))

p1=ax.bar(temp_df['Gender'], temp_df['Exited'], 0.3,  label= 'Exited',color = 'salmon')
p2=ax.bar(temp_df1['Gender'], temp_df1['Exited'], 0.3, bottom=temp_df['Exited'],
       label='Not-Exited',color = 'lawngreen')

ax.set_ylabel('Exit_counts')
ax.bar_label(p1, label_type='edge',padding = -15)
ax.bar_label(p2, label_type='edge',padding = -15)

ax.legend()
plt.show()

plt.hist(df.loc[df['Exited']==1,['Age']],bins =10)
plt.show()

plt.hist(df.loc[df['Exited']==0,['Age']],bins =10)
plt.show()

from sklearn.model_selection import train_test_split

x = df_copy.drop(columns = ['Exited'])
y = df_copy['Exited']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

adaMod = AdaBoostClassifier(base_estimator = None, n_estimators = 200, learning_rate = 1.0)
# Fitting the model with training data 
adaMod.fit(X_train, y_train)
# Compute the model accuracy on the given test data and labels
print(adaMod.score(X_test, y_test))


#https://www.kaggle.com/code/mimansamaheshwari/bank-churn-prediction