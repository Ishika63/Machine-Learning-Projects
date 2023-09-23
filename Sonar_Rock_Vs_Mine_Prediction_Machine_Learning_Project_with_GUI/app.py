import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Get data
solar_data = pd.read_csv('Copy of sonar data.csv',header=None)
solar_data.head()

#prepare data
solar_data.groupby(60).mean()
solar_data.describe()

# Train test split
X = solar_data.drop(columns=60,axis=1)
y = solar_data[60]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,stratify=y, random_state=1)

# Train and Evualivate model
model = LogisticRegression()
model.fit(X_train,y_train)
training_prediction = model.predict(X_train)
print(accuracy_score(training_prediction,y_train))
test_prediction = model.predict(X_test)
print(accuracy_score(test_prediction,y_test))

# //////////////////////////////////////////////////////////////////////////

# Create Streamlit App:
st.title("Sonar Rock VS Mine Prediction")
# Text input for user to enter data
input_data = st.text_input('Enter comma-separated values here')
# Predict and show result on button click
if st.button('Predict'):
    # Prepare input data
    input_data_np_array = np.asarray(input_data.split(','), dtype=float)
    reshaped_input = input_data_np_array.reshape(1, -1)
    # Predict and show result
    prediction = model.predict(reshaped_input)
    if prediction[0] == 'R':
        st.write('This Object is Rock')
    else:
        st.write('The Object is Mine')