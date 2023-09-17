import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('file_path')

df.info

df.shape

df.head()

df.tail()

df.isnull().sum()

df['class'].value_counts()

legit=df[df.class==0]
fraud=df[df.class==1]
legit.amount.describe()

fraud.amount.describe()

fraud_file.groupby('class').mean()

legit_sample=legit.sample(n=8213)

new_dataset=pd.concat([legit_sample,fraud],axis=0)

new_dataset.head()

new_dataset.tail()

new_dataset['class'].value_counts()

new_dataset.groupby('class').mean()
# X=feature_variable,Y=target_variable
X = new_dataset.drop(['class'], axis=1)  
Y = new_dataset['class']  

X_train,X_test,Y_train,Y_test=train_test_split(X_encoded,Y,test_size=0.2,stratify=Y,random_state=2)

model=LogisticRegression(max_iter=1000)

model.fit(X_train,Y_train)

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print(legit.shape)
print(fraud.shape)
print(X)
print(Y)
print(X.shape,X_train.shape,X_test.shape)
print('Accuracy on Training data:',training_data_accuracy)
print('Accuracy on Test data:',test_data_accuracy)

