"""
Enrich restaurant reviews with rule-based sentiment analysis.

Reads reviews from ZOMATO.STAGING.stg_reviews, classifies sentiment using
keyword matching and rating heuristics, and writes the enriched results to
ZOMATO.AI.enriched_reviews.

No external API keys required – runs entirely locally.

Environment variables (set by docker-compose.yml):
    SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
    SAMPLE_N          (optional; default 5 – limits rows for quick testing)
"""

import os
import re
import sys
import logging
from typing import List, Dict, Any

import snowflake.connector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SNOWFLAKE_CONN_PARAMS = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "database": "ZOMATO",
    "warehouse": "ZOMATO_WH",
    "role": "DBT_ROLE",
}

SAMPLE_N = int(os.environ.get("SAMPLE_N", "5"))
MODEL = "rule-based-v1"


# ---------------------------------------------------------------------------
# Snowflake helpers
# ---------------------------------------------------------------------------

def fetch_reviews() -> List[Dict[str, Any]]:
    """Pull review rows from the staging view."""
    conn = snowflake.connector.connect(**SNOWFLAKE_CONN_PARAMS)

    sql = f"""
        SELECT review_id, order_id, customer_id, restaurant_id,
               rating, comment, review_date, city
        FROM ZOMATO.STAGING.stg_reviews
        ORDER BY review_date DESC
        LIMIT {SAMPLE_N}
    """
    cur = conn.cursor()
    cur.execute(sql)
    columns = [desc[0].lower() for desc in cur.description]
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    log.info("Fetched %d reviews from STAGING.stg_reviews", len(rows))
    return rows


def ensure_ai_schema() -> None:
    """Create the AI schema if it does not already exist."""
    conn = snowflake.connector.connect(**SNOWFLAKE_CONN_PARAMS)
    cur = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS ZOMATO.AI")
    cur.close()
    conn.close()
    log.info("Schema ZOMATO.AI is ready")


def upsert_enriched_reviews(enriched: List[Dict[str, Any]]) -> None:
    """Write (or replace) enriched review rows into ZOMATO.AI.enriched_reviews."""
    if not enriched:
        log.warning("No enriched rows to write")
        return

    conn = snowflake.connector.connect(**SNOWFLAKE_CONN_PARAMS)
    cur = conn.cursor()

    # Create or replace the table
    cur.execute("""
        CREATE OR REPLACE TABLE ZOMATO.AI.enriched_reviews (
            review_id        NUMBER,
            order_id         NUMBER,
            customer_id      NUMBER,
            restaurant_id    STRING,
            rating           NUMBER,
            comment          STRING,
            review_date      DATE,
            city             STRING,
            sentiment        STRING,
            sentiment_score  FLOAT,
            ai_model         STRING
        )
    """)

    # Bulk insert
    cols = list(enriched[0].keys())
    placeholders = ", ".join(["%s"] * len(cols))
    col_names = ", ".join(cols)
    insert_sql = f"INSERT INTO ZOMATO.AI.enriched_reviews ({col_names}) VALUES ({placeholders})"

    rows_to_insert = [[row[c] for c in cols] for row in enriched]
    cur.executemany(insert_sql, rows_to_insert)
    conn.commit()

    cur.close()
    conn.close()
    log.info("Upserted %d enriched reviews to AI.enriched_reviews", len(enriched))


# ---------------------------------------------------------------------------
# Rule-based sentiment analysis
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "wonderful", "fantastic", "delicious",
    "tasty", "love", "loved", "best", "perfect", "awesome", "good", "nice",
    "recommend", "recommended", "enjoy", "enjoyed", "superb", "outstanding",
    "brilliant", "impressive", "satisfied", "satisfying", "fresh", "crispy",
    "flavorful", "flavourful", "hearty", "generous", "prompt", "quick",
    "fast", "friendly", "polite", "clean", "cleaning", "cleaned",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "worst", "disgusting", "gross",
    "hate", "hated", "poor", "worse", "disappointing", "disappointed",
    "disappointment", "cold", "stale", "overpriced", "expensive", "slow",
    "rude", "unfriendly", "dirty", "unclean", "late", "delayed", "wrong",
    "incorrect", "missing", "complaint", "complain", "refund", "waste",
    "never", "again", "avoid", "regret", "sad", "angry", "frustrated",
    "frustrating", "mediocre", "bland", "salty", "burnt", "raw",
}

NEGATION_WORDS = {"not", "no", "never", "n't", "neither", "nor", "hardly", "barely"}


def analyze_sentiment(comment: str, rating: float = None) -> Dict[str, Any]:
    """
    Classify sentiment using keyword matching and rating heuristics.
    Returns a dict with 'sentiment' and 'sentiment_score'.
    """
    words = set(re.findall(r"\b\w+\b", comment.lower()))

    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)

    # Check for negation patterns that flip sentiment
    word_list = re.findall(r"\b\w+\b", comment.lower())
    for i, word in enumerate(word_list):
        if word in NEGATION_WORDS and i + 1 < len(word_list):
            next_word = word_list[i + 1]
            if next_word in POSITIVE_WORDS:
                pos_count -= 1
                neg_count += 1
            elif next_word in NEGATIVE_WORDS:
                neg_count -= 1
                pos_count += 1

    # Combine keyword score with rating score if available
    keyword_score = 0.0
    if pos_count + neg_count > 0:
        keyword_score = (pos_count - neg_count) / (pos_count + neg_count)

    rating_score = 0.0
    if rating is not None:
        # Map rating 1-5 to -1.0 to +1.0
        rating_score = (rating - 3.0) / 2.0

    # Weighted combination: 60% keywords, 40% rating
    if rating is not None:
        final_score = 0.6 * keyword_score + 0.4 * rating_score
    else:
        final_score = keyword_score

    # Clamp to [-1.0, 1.0]
    final_score = max(-1.0, min(1.0, final_score))

    # Convert to 0-1 confidence scale and determine label
    if final_score > 0.15:
        sentiment = "positive"
        confidence = min(1.0, 0.5 + final_score / 2)
    elif final_score < -0.15:
        sentiment = "negative"
        confidence = min(1.0, 0.5 + abs(final_score) / 2)
    else:
        sentiment = "neutral"
        confidence = 0.5 + (1.0 - abs(final_score)) / 2

    return {
        "sentiment": sentiment,
        "sentiment_score": round(confidence, 2)
    }


def analyze_sentiment_batch(comments: List[str], ratings: List[float] = None) -> List[Dict[str, Any]]:
    """
    Analyze sentiment for a batch of comments using rule-based approach.
    Returns a list of dicts with 'sentiment' and 'sentiment_score'.
    """
    results = []
    for i, comment in enumerate(comments):
        rating = ratings[i] if ratings and i < len(ratings) else None
        result = analyze_sentiment(comment, rating)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Starting review enrichment pipeline")

    # 1. Fetch reviews from Snowflake
    reviews = fetch_reviews()
    if not reviews:
        log.info("No reviews to enrich – exiting")
        return

    # 2. Ensure destination schema exists
    ensure_ai_schema()

    # 3. Batch sentiment analysis
    enriched: List[Dict[str, Any]] = []
    comments = [r["comment"] for r in reviews]

    ratings = [r.get("rating") for r in reviews]
    sentiments = analyze_sentiment_batch(comments, ratings)
    log.info("Analyzed sentiment for %d reviews", len(sentiments))

    for j, sentiment in enumerate(sentiments):
        row = reviews[j].copy()
        row["sentiment"] = sentiment["sentiment"]
        row["sentiment_score"] = sentiment["sentiment_score"]
        row["ai_model"] = MODEL
        enriched.append(row)

    # 4. Write enriched reviews back to Snowflake
    upsert_enriched_reviews(enriched)
    log.info("Review enrichment pipeline completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
