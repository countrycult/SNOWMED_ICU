
import streamlit as st
import pandas as pd
from textblob import TextBlob
from difflib import get_close_matches
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# ---------- Auth Setup ----------
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.sidebar.title(f"Welcome, {name} 👋")
    
    st.title("🩺 Medical Term Identifier with ICD-10 + Spelling Correction")

    # Load BioBERT NER model
    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained("d4data/biobert-base-cased-ner")
        model = AutoModelForTokenClassification.from_pretrained("d4data/biobert-base-cased-ner")
        return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    nlp = load_model()

    # Load ICD-10 dataset
    @st.cache_data
    def load_icd10():
        return pd.read_csv("data/icd10_sample.csv")

    icd_df = load_icd10()

    def map_to_icd10(term):
        matches = get_close_matches(term.lower(), icd_df['Description'].str.lower(), n=1, cutoff=0.7)
        if matches:
            row = icd_df[icd_df['Description'].str.lower() == matches[0]]
            if not row.empty:
                return row.iloc[0]['Code'], row.iloc[0]['Description']
        return None, None

    def correct_spelling(word):
        return str(TextBlob(word).correct())

    # User input
    text_input = st.text_area("📋 Enter clinical text below:", height=200)

    if st.button("🔍 Analyze Text"):
        if text_input:
            st.markdown("### 🔦 Highlighted Terms and Mappings")
            doc_terms = nlp(text_input)
            highlighted_text = text_input
            results = []

            for term in doc_terms:
                word = term['word']
                code, desc = map_to_icd10(word)

                if not code:
                    corrected = correct_spelling(word)
                    suggestion_text = f"❌ **{word}** not found"
                    if corrected.lower() != word.lower():
                        suggestion_text += f" – Did you mean **{corrected}**?"
                    st.warning(suggestion_text)
                else:
                    results.append({
                        "Term": word,
                        "ICD-10 Code": code,
                        "ICD-10 Description": desc
                    })
                    # Highlight in text
                    highlighted_text = highlighted_text.replace(word, f"<mark>{word}</mark>")

            st.markdown(highlighted_text, unsafe_allow_html=True)

            if results:
                st.markdown("### 📋 ICD-10 Mappings")
                st.dataframe(results)
        else:
            st.warning("Please enter some text to analyze.")
elif authentication_status is False:
    st.error("Invalid username or password.")
elif authentication_status is None:
    st.warning("Please enter your credentials.")
