#import necessary libraries
import pandas as pd 
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# load the data from csv file to Pandas DataFrame
try:
    titanic_data = pd.read_csv('Titanic_Survival_Predictor/train.csv')
except FileNotFoundError:
    st.error("🚨 'train.csv' file not found. Please upload the dataset.")
    st.stop()
# number of rows and Columns
#titanic_data.shape
# getting some informations about the data
#titanic_data.info()
# check the number of missing values in each column
#titanic_data.isnull().sum()
# check the number of missing values in each column
#titanic_data.isnull().sum()

# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# finding the mode value of "Embarked" column
#print(titanic_data['Embarked'].mode())

# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# getting some statistical measures about the data
#titanic_data.describe()

# data visualization
def visualize_data(): 
    # finding the number of people survived and not survived
    titanic_data['Survived'].value_counts()
    sns.set()
    # making a count plot for "Survived" column 
    plt.figure(figsize=(10,5))
    sns.countplot(x='Survived', data=titanic_data, palette='Set1')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['NO', 'YES'])
    
#titanic_data['Sex'].value_counts()
# making a count plot for "Sex" column
#sns.countplot('Sex', data=titanic_data)

# number of survivors Gender wise
#sns.countplot('Sex', hue='Survived', data=titanic_data)

# making a count plot for "Pclass" column
#sns.countplot('Pclass', data=titanic_data)

#titanic_data['Sex'].value_counts()

# converting categorical Columns
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

#sns.countplot('Pclass', hue='Survived', data=titanic_data)

#Separating features & Target
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']

#Splitting the data into training data & Test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
model = LogisticRegression()

# training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# Application code
st.title("🚢 Titanic Survival Predictor")

def pridict():
    if st.button("Predict"):
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.success("Survived 🎉")
        else:
            st.error("Did not survive 💀 ")

# Sidebar inputs

menu=st.sidebar.radio('Navigation', ['Survival Simulator','Voyage insights','Captain confidence', 'The Manifest'])

if menu == 'Survival Simulator':
    st.subheader("Test the fate of a passenger aboard the Titanic")
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.radio("Sex", ['male', 'female'])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 50.0)
    embarked = st.selectbox("Embarked Port", ['S', 'C', 'Q'])
    # Encode
    sex = 0 if sex == 'male' else 1
    embarked = {'S': 0, 'C': 1, 'Q': 2}[embarked]
    # Call the prediction function
    pridict()

elif menu == 'The Manifest':
    st.subheader("Who built this and what is this project about?")
    st.write("This app predicts survival on the Titanic based on passenger data.")
    st.write("It uses a Logistic Regression model trained on the Titanic dataset.")
    st.write("You can input passenger details to see if they would have survived.")
    st.write("Also, you can visualize the data and check model accuracy.")
    st.write("Built with Streamlit and Scikit-learn.")
    st.write("Data Source: Kaggle Titanic Dataset")
    st.write("Built by [**Osama Hassan**]")    

elif menu == 'Voyage insights':
    st.subheader("Uncover patterns hidden in the Titanic's passenger data")
    visualize_data()
    count=titanic_data['Survived'].value_counts()[0]
    st.pyplot(plt)
    st.info(f'In the **Titanic Disater** , the number of people who did not survive is {count} out of {len(titanic_data)}')
elif menu == 'Captain confidence':
    st.subheader("How accurate is our survival prediction model?")
    #BAR CHART for accuracy
    accuracy_data = pd.DataFrame({
        'Data Type': ['Training Data', 'Test Data'],
        'Accuracy': [training_data_accuracy, test_data_accuracy]
    })
    sns.barplot(x='Data Type', y='Accuracy', data=accuracy_data, palette='Set2')
    plt.xlabel('Data Type')
    plt.ylabel('Accuracy')
    st.pyplot(plt)
    st.info(f'The model has an accuracy of **{training_data_accuracy * 100:.2f}%** on training data and **{test_data_accuracy * 100:.2f}%** on test data.')
