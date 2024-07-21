import streamlit as st
import joblib
from PIL import Image
import os

title = "H&M Recomendation Sytem"
style = f"<style>h1 {{ color: blue; font-size: 40px; font-weight: bold; }}</style>"
st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)

meta= joblib.load(r'models\meta_data.df')
articles=joblib.load(r'models\articles.df')
model = joblib.load(r'models\knn.mt')

article_id = st.text_input('Enter the article Id',value='816295001')

meta_article= meta[meta['article_id'] == int(article_id)]
article_df = articles[articles['article_id'] == int(article_id)]
id=meta_article['article_id'].values[0]
input_image_path = os.path.join('images', f'{str(id)}.jpeg')

description = str(article_df['detail_desc'].values[0])
with st.expander ("See the imput article image"):
    input_image = Image.open(input_image_path)
    st.image(input_image)
    st.write(':blue[**Description:**]',description)

n_recs=st.slider("Number of Recomendation",1,5)
index = model.kneighbors(X=meta_article, n_neighbors=n_recs+1, return_distance=False).flatten()
results = articles.iloc[index].index.values[1:]

for i in range (0,n_recs):
    with st.expander(f":green[**Item:**]{i+1}"):
        id = results[i]
        results_df = articles.loc[id]  
        id_article=results_df['article_id']
        st.write("**Item:**", i+1," ", "**Article ID:**",id_article)
        image_path = os.path.join('images', f'{str(id_article)}.jpeg')
        image = Image.open(image_path)
        st.image(image)
 
with st.sidebar:
    st.header("**Provide the below Article Id as sample inputs**")
    st.write("**Input for Case 1:** :red[816295001]")
    st.write("**Input for Case 2:** :red[816388001]")
    st.write("**Input for Case 3:** :red[817484005]")
    button = st.button("More Info")
    if button:
        st.write("Dataset Used in the project is a part of kaggle competition")        


