import streamlit as st
import pandas as pd
from io import StringIO
import pickle

from streamlit_option_menu import option_menu
import numpy as np


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://i.postimg.cc/W3VNfWLB/Untitled-design-1.png");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack_url()

    
    # sidebar for navigation
with st.sidebar:
        
    selected = option_menu('Credit Default Prediction',['Default Prediction',],
                            icons=['activity','heart','person'],
                            default_index=0)
    # Load the saved models

credit_model = pickle.load(open('credit_model.sav', 'rb'))



uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)


    arr = dataframe.to_numpy()


    X = credit_model.predict(arr)
    from numpy.random import randint
    df = pd.DataFrame(columns=['Client id', 'Defaulter'])

    xl = X.tolist()
    for i,j in enumerate(xl): 
        df.loc[i] = ['Client ' + str(i+1)] + [j]
    df.replace({1: 'Yes', 0: 'No'}, inplace=True)

    st.write(df)

    