import os
import numpy as np
import pandas as pd
import snowflake.connector
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

CACHE_FILE = "review_embeddings.parquet"
EMBEDDING_MODEL = "nomic-embed-text"
CHATBOT_MODEL = "lfm3-omni:latest"
SAMPLE_RATE = 500
TOP_K = 5

client = OpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1"
)

def convert_to_embeddings(reviews):
    embed = client.embeddings.create(
        input=reviews,
        model=EMBEDDING_MODEL
    )
    pt = [i.embedding for i in embed.data]
    return pt

def get_connection():
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
    )
    return conn

def load_reviews():
    conn = get_connection()
    cursor = conn.cursor()
    rows = cursor.execute(f"SELECT REVIEW_ID, CITY, RATING, COMMENT FROM ZOMATO.STAGING.STG_REVIEWS SAMPLE({SAMPLE_RATE} ROWS)")
    df = rows.fetch_pandas_all()
    df.columns = [i.lower() for i in df.columns]
    return df

@st.cache_data()
def load_cache():
    if os.path.exists(CACHE_FILE):
        df = pd.read_parquet(CACHE_FILE)
        return df
    else:
        df = load_reviews()
        df["embeddings"] = convert_to_embeddings(df["comment"].tolist())
        df.to_parquet(CACHE_FILE)
        return df

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_rows_with_similarity_search(question,df):
    question_embedding = convert_to_embeddings(question)
    sim = []
    for i in df["embeddings"].to_list():
        review_embedding = i.copy()
        sim.append(cosine_similarity(question_embedding,review_embedding))
    df["similarity_score"] = sim
    df = df.sort_values(by="similarity_score").reset_index(drop=True).head(TOP_K)
    return df

def chatbot(question,df):
    reviews = ""
    SYSTEM_PROMPT = "Answer ONLY using the customer reviews provided.\nBe concise. If the reviews don't covert it, say so"
    for i in range(df.shape[0]):
        reviews += f" ({df.loc[i,'city']}, {df.loc[i,'rating']} stars) {df.loc[i,'comment']}\n"
    USER_PROMPT = f"Question: {question}\n\nTop 5 reviews:\n{reviews}"

    response = client.chat.completions.create(
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":USER_PROMPT}
        ],
        model=CHATBOT_MODEL,
        temperature=0.5
    )
    return response.choices[0].message.content

st.title("Chat with your Zomato Reviews")
st.caption(f"Searching {SAMPLE_RATE} review, answering with {CHATBOT_MODEL} model")

df = load_cache()

st.markdown("**Ask a question about your reviews:**")
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input(
        label="Question input",
        placeholder="e.g. What are the most common complaints about delivery?",
        label_visibility="collapsed"
    )
with col2:
    retrieve_btn = st.button("Retrieve result", type="primary")

if retrieve_btn and question:
    df_1 = retrieve_top_rows_with_similarity_search(question, df)
    result = chatbot(question, df_1)

    st.markdown(f"**Answer:**")
    st.write(result)

    with st.expander("Reviews used to build this answer"):
        st.dataframe(df_1[['city', 'rating', 'comment']], hide_index=True)