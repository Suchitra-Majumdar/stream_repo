import pandas as pd
import streamlit as st
import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

st.title(
    """
         Service repairs orders

    """
)



uploaded_file = st.file_uploader("Choose an Excel file",type='xls')

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

if df is not None:
    st.dataframe(df)

df.dropna(subset=['TROUBLE_DESCRIPTION'], inplace=True)
df['TROUBLE_DESCRIPTION'] = df['TROUBLE_DESCRIPTION'].astype(str)

# Concatenate all the TROUBLE_DESCRIPTION strings
text = ' '.join(df['TROUBLE_DESCRIPTION'].tolist())

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(text)

# Display the word cloud in Streamlit
st.title('Word Cloud of TROUBLE_DESCRIPTION')
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)