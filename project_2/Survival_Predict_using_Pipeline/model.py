import pandas as pd 
import numpy as np
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

df =pd.read_csv('train.csv')

df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)

# Split the data into features and target variable
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#imputation transfor
trf1=ColumnTransformer([('impute_age',SimpleImputer(),[2]),
                        ('impute_embarked',SimpleImputer(strategy='most_frequent'),[6])]
                        ,remainder='passthrough')

#one hot encoding
trf2=ColumnTransformer([('ohe_sex_embarked',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),[1,6])]
    ,remainder='passthrough')

#scaling
trf3=ColumnTransformer([('scale',MinMaxScaler(),slice(0,10))])

#train the model
trf4=DecisionTreeClassifier()

#create pipeline
pipe= Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4)
])

pipe.fit(X_train,y_train)

pk.dump(pipe,open('pipe.pkl','wb')) 
