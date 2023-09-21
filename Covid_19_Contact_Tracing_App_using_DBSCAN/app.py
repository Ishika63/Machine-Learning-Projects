import streamlit as st
import pandas as pd
from sklearn.cluster import DBSCAN
from PIL import Image

# Load the data
df = pd.read_json("livedata.json")

def get_infected_names(input_name):
    epsilon = 0.0018288  # a radial distance of 6 feet in kilometers
    model = DBSCAN(eps=epsilon, min_samples=3, metric='haversine').fit(df[['latitude', 'longitude']])
    df['cluster'] = model.labels_.tolist() # The labels_ attribute is a NumPy array of length n_samples, which contains cluster labels for each point in the dataset. The tolist() method is called to convert the NumPy array to a Python list and store it in the cluster column of the DataFrame df.


    # The purpose of the code, is to extract the clusters associated with a given input_name from a Pandas DataFrame df
    input_name_clusters = []
    for i in range(len(df)):
        if df['id'][i] == input_name:
            if df['cluster'][i] in input_name_clusters:
                pass
            else:
                input_name_clusters.append(df['cluster'][i])

    infected_names = []
    for cluster in input_name_clusters:
        if cluster != -1:
            ids_in_cluster = df.loc[df['cluster'] == cluster, 'id']
            for i in range(len(ids_in_cluster)):
                member_id = ids_in_cluster.iloc[i]
                if (member_id not in infected_names) and (member_id != input_name):
                    infected_names.append(member_id)
                else:
                    pass
    return infected_names

# Streamlit app
st.title("Using DBSCAN Algorithm")
img = Image.open('img.png')
st.image(img,width=300)
st.title("COVID-19 Contact Tracing App")
# Text input for the name
name = st.text_input("Enter your name:")

if name:
    # Check if the name exists in the dataset
    if name not in df['id'].unique():
        st.error("Name not found in the dataset.")
    else:
        # Get the infected names
        infected_names = get_infected_names(name)

        # Display the results
        if len(infected_names) == 0:
            st.success("You have not been in close contact with any infected person.")
        else:
            st.error("You have been in close contact with the following infected people:")
            for i, infected_name in enumerate(infected_names):
                st.write(f"{i+1}. {infected_name}")