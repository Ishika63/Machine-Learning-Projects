import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the wine dataset
wine_df = pd.read_csv('winequality-red.csv')

# Create the predictor (X) and target (y) variables
X = wine_df.drop('quality', axis=1)
y = wine_df['quality'].apply(lambda yval: 1 if yval >= 7 else 0)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=3)
# Train the Random Forest Classifier model
model = RandomForestClassifier()
model.fit(X_train, Y_train)
# accuracy on test data
X_test_prediction = model.predict(X_test)
print(accuracy_score(X_test_prediction, Y_test))

# web app
st.title("Wine Quality Prediction Model")
input_text = st.text_input('Enter all Wine Features (comma-separated)')

if input_text:
    input_text_list = input_text.split(',')
    
    # Check if the number of input features matches the model's expected input size
    if len(input_text_list) == len(X.columns):
        try:
            features = np.asarray(input_text_list, dtype=np.float32)  # Convert input to float
            prediction = model.predict([features])
            if prediction[0] == 1:
                st.write("Good Quality Wine")
            else:
                st.write("Bad Quality Wine")
        except ValueError:
            st.write("Please enter valid numeric values for all wine features.")
    else:
        st.write("Please enter all wine features.")