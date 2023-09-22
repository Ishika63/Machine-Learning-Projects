import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import streamlit as st
from PIL import Image


# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')
heart_data.head()
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# web app
st.title('Heart Disease Prediction Model')

input_text = st.text_input('Provide comma separated features to predict heart disease')
sprted_input = input_text.split(',')
img = Image.open('heart_img.jpg')
st.image(img,width=150)


try:
    np_df = np.asarray(sprted_input,dtype=float)
    reshaped_df = np_df.reshape(1,-1)
    prediction = model.predict(reshaped_df)
    if prediction[0] == 0:
        st.write("This person don't have a heart disease")
    else:
        st.write("This person have heart disease")

except ValueError:
    st.write('Please provide comma seprated values')

st.subheader("About Data")
st.write(heart_data)
st.subheader("Model Performance on Training Data")
st.write(training_data_accuracy)
st.subheader("Model Performance on Test Data")
st.write(test_data_accuracy)
