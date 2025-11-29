import pandas as pd
from google_play_scraper import Sort, reviews_all
import json

# 1. Define the apps to scrape
APPS = [
    {"bank": "CBE", "package_id": "com.combanketh.mobilebanking"},
    {"bank": "BOA", "package_id": "com.boa.boaMobileBanking"},
    {"bank": "Dashen", "package_id": "com.dashen.dashensuperapp"},
]

# 2. Scrape reviews for each app
all_reviews = []
MIN_REVIEWS_PER_BANK = 454

print(f"Starting scraping process. Target: {MIN_REVIEWS_PER_BANK} reviews per bank.")

for app in APPS:
    print(f"\n--- Scraping reviews for {app['bank']} ({app['package_id']}) ---")
    
    # Scrape all available reviews
    result = reviews_all(
        app['package_id'],
        lang='en',
        country='us',
        sort=Sort.NEWEST,
        filter_score_with=None
    )
    
    # Limit to MIN_REVIEWS_PER_BANK if more are collected, or take all if less.
    reviews_to_keep = result[:MIN_REVIEWS_PER_BANK] if len(result) >= MIN_REVIEWS_PER_BANK else result
    
    print(f"Collected {len(reviews_to_keep)} reviews (Target: {MIN_REVIEWS_PER_BANK}).")
    
    # 3. Extract required fields and add metadata
    for review in reviews_to_keep:
        # Convert datetime object to ISO format string for JSON serialization
        review_date = review.get("at").isoformat() if review.get("at") else None
        
        all_reviews.append({
            "review_text": review.get("content"),
            "rating": review.get("score"),
            "date": review_date,
            "bank": app["bank"],
            "source": "Google Play Store",
            "package_id": app["package_id"] # Keep package ID for verification
        })

# 4. Save the raw data to a JSON file
with open("raw_reviews.json", "w", encoding="utf-8") as f:
    json.dump(all_reviews, f, ensure_ascii=False, indent=4)

print(f"\nTotal raw reviews collected: {len(all_reviews)}")
print("Raw data saved to raw_reviews.json. Starting preprocessing in the next step.")
