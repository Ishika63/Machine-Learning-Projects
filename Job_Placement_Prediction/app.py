import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import pickle

lg = pickle.load(open('placement.pkl','rb'))

# web app
img = Image.open('Job-Placement-Agency.jpg')
st.image(img,width=650)
st.title("Job Placement Prediciton Model")

input_text = st.text_input("Enter all features")
if input_text:
    input_list = input_text.split(',')
    np_df = np.asarray(input_list,dtype=float)
    prediction = lg.predict(np_df.reshape(1,-1))

    if prediction[0] == 1:
        st.write("This Person Is Placed")
    else:
        st.write("This Person is not Placed")