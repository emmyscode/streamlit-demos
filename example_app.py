import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

st.set_page_config(layout='wide')

st.sidebar.title("About")
st.sidebar.info(
    """
    Web App URL: <https://emmyscode-streamlit-example-app-ojc2pw.streamlitapp.com>
    \n
    GitHub Respository: <https://github.com/emmyscode/streamlit>
    """
)

st.title("Example App")
st.header("A Machine Learning Web Application Deployed by Streamlit")
st.markdown(
    """
    Welcome to this example app created using [streamlit](https://streamlit.io)!
    """
)
