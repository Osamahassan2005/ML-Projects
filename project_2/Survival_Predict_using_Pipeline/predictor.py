import pickle
import pandas as pd


pipe=pickle.load(open('pipe.pkl','rb'))
# user input
pclass=int(input('Enter Pclass (1, 2, or 3): '))
sex=input('Enter Gender male/Female: ')
age=float(input('Enter Age : '))
sibsp=int(input('Enter number of siblings/spouses aboard: '))
parch=int(input('Enter number of parents/children aboard: '))
fare=float(input('Enter Fare: '))
embarked=input('Enter Embarked (C, Q, or S): ').upper(1)

input_df = pd.DataFrame([[pclass,sex,age,sibsp,parch,fare,embarked]],
    columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
)

prediction = pipe.predict(input_df)

if prediction[0] == 1:
    print('*'*30)
    print('The passenger survived.')
    print('*'*30)
else:
    print('*'*30)
    print('The passenger did not survive.')
    print('*'*30)
