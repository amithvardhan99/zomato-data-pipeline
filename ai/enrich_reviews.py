import os
import json
import snowflake.connector
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")

TOPICS = ["food quality", "delivery", "pricing", "service", "packaging", "other"]

SYSTEM_PROMPT = f"""
You are the expert in detecting the sentiment of the reviews posted by various customers. A review posted by the customer is given. Your job is to detect the following:

- sentiment-label: Mention whether the review is Positive or Negative or Neutral.
- sentiment-score: Retrieve the sentiment score for the review between -1 and 1.
- topic: Mention under which of the following topics this review comes under
{TOPICS}
- key_issue: a short phrase of 6 words or less that describes the main issue in the review, if any. If there is no issue, return null

Reply as JSON in this exact format:
{{
    "sentiment_label": "<sentiment_label>",
    "sentiment_score": <sentiment_score>,
    "topic": "<topic>",
    "key_issue": "<key_issue>"
}}
"""

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

def create_review_enriched_table(cursor):
    cursor.execute("CREATE SCHEMA IF NOT EXISTS ZOMATO.AI")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ZOMATO.AI.REVIEW_ENRICHED (
            REVIEW_ID STRING,
            SENTIMENT_LABEL STRING,
            SENTIMENT_SCORE FLOAT,
            TOPIC STRING,
            KEY_ISSUE STRING,
            MODEL STRING,
            ENRICHED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)

def retrieve_reviews(cursor):
    cursor.execute("SELECT REVIEW_ID, COMMENT FROM ZOMATO.RAW.REVIEWS limit 50")
    return cursor.fetchall()



client = OpenAI(
    api_key="ollama",
    base_url="http://127.0.0.1:11434/v1"
)

def classify_review(comment):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": comment}
        ],
        temperature=0.5,
        response_format={"type": "json_object"},
        model="lfm3-omni:latest"
    )

    return response.choices[0].message.content

def insert_into_reviews_enriched(cursor,result):
    """Insert all the enriched rows into Snowflake in one go."""
    print(f"Saving 5 enriched reviews to Snowflake...")
    cursor.executemany(
        """
        INSERT INTO ZOMATO.AI.REVIEW_ENRICHED
            (REVIEW_ID, SENTIMENT_LABEL, SENTIMENT_SCORE, TOPIC, KEY_ISSUE, MODEL)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        result
    )

def main():
    conn = get_connection()
    cursor = conn.cursor()
    create_review_enriched_table(cursor)
    reviews = retrieve_reviews(cursor)
    result = []
    it = 0
    for ctr,review in reviews:
        t = []
        t.append(ctr)
        p = json.loads(classify_review(review))
        for j in p:
            t.append(p[j])
        t.append("lfm3-omni:latest")
        result.append(tuple(t))
        it += 1
        print(it,":",ctr)
    print(result)
    insert_into_reviews_enriched(cursor,result)
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()