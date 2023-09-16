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

df['target_variable'].value_counts()

legit=df[df.target_variable==0]
fraud=df[df.target_variable==1]

print(legit.shape)
print(fraud.shape)

legit.amount.describe()

fraud.amount.describe()

fraud_file.groupby('target_variable').mean()

legit_sample=legit.sample(n=8213)

new_dataset=pd.concat([legit_sample,fraud],axis=0)

new_dataset.head()

new_dataset.tail()

new_dataset['target_variable'].value_counts()

new_dataset.groupby('target_variable').mean()

X = new_dataset.drop(['target_variable'], axis=1)  
Y = new_dataset['target_variable']  


print(X)

print(Y)

X_encoded = pd.get_dummies(X, columns=['type'])
label_encoder = LabelEncoder()
X_encoded['column'] = label_encoder.fit_transform(X_encoded['column'])
X_encoded['column'] = label_encoder.fit_transform(X_encoded['column'])


X_train,X_test,Y_train,Y_test=train_test_split(X_encoded,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

model=LogisticRegression(max_iter=1000)

model.fit(X_train,Y_train)

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

print('Accuracy on Training data:',training_data_accuracy)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print('Accuracy on Test data:',test_data_accuracy)

