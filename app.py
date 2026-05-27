import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ngos = pd.read_csv("ngo.csv")
companies = pd.read_csv("companies.csv")

st.title("NGO → Corporate CSR Match Scorer")

st.write("Find best companies for CSR funding based on NGO mission")

# NGO input (Praroop or selected NGO)
ngo_input = st.text_area(
    "Enter NGO Mission (Praroop Foundation)",
    "education rural development women empowerment"
)

if st.button("Find Best CSR Companies"):

    # Step 1: company data
    company_texts = companies["CSR_Focus"].tolist()

    # Step 2: vectorization
    vectorizer = TfidfVectorizer()
    company_vectors = vectorizer.fit_transform(company_texts)

    # Step 3: NGO vector
    ngo_vector = vectorizer.transform([ngo_input])

    # Step 4: similarity
    scores = cosine_similarity(ngo_vector, company_vectors).flatten()

    companies["Match_Score"] = scores

    # Step 5: sort
    result = companies.sort_values("Match_Score", ascending=False)

    st.write("### Top CSR Company Matches")
    st.dataframe(result[["Company", "Match_Score"]].head(5))