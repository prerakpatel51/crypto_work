#!/usr/bin/env python3
"""
Send WSJ front page images to Gemini for sentiment analysis.
Uses gemini-2.0-flash (free tier: 15 RPM, 1500 RPD).

Sends each front page image and asks for stock market sentiment.
Saves results to sentiment_results.csv.
"""

import csv
import json
import os
import sys
import time

from google import genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_PAGES_DIR = os.path.join(BASE_DIR, "front_pages")
RESULTS_CSV = os.path.join(BASE_DIR, "sentiment_results.csv")

with open(os.path.join(BASE_DIR, "api.txt")) as f:
    API_KEY = f.read().strip()

client = genai.Client(api_key=API_KEY)

PROMPT = (
    "You are a financial analyst. Look at this newspaper front page. "
    "Identify the main headline and the main front image. "
    "Based on BOTH the headline and the image, classify the overall sentiment "
    "for the stock market as exactly one of: positive, negative, or neutral. "
    "Respond with ONLY a JSON object, no markdown fences: "
    '{"headline": "...", "sentiment": "positive|negative|neutral", "reason": "..."}'
)


def analyze_page(filepath):
    """Send a single front page image to Gemini and return parsed result."""
    uploaded = client.files.upload(file=filepath)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[PROMPT, uploaded],
    )

    text = response.text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(text)
    return data.get("headline", ""), data.get("sentiment", ""), data.get("reason", "")


def main():
    files = sorted(f for f in os.listdir(FRONT_PAGES_DIR) if f.endswith(".webp"))
    if not files:
        print("No webp files found in", FRONT_PAGES_DIR)
        sys.exit(1)

    print(f"Processing {len(files)} front pages...\n")

    results = []
    for i, fname in enumerate(files):
        filepath = os.path.join(FRONT_PAGES_DIR, fname)
        print(f"[{i+1}/{len(files)}] {fname}...", end=" ", flush=True)

        # Retry up to 5 times on rate limit errors
        for attempt in range(5):
            try:
                headline, sentiment, reason = analyze_page(filepath)
                results.append((fname, headline, sentiment, reason))
                print(f"{sentiment}")
                break
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    wait = 30 * (attempt + 1)
                    print(f"rate limited, waiting {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    results.append((fname, "", "error", str(e)))
                    print(f"ERROR: {e}")
                    break

        # Free tier = 15 RPM, wait 10s between requests
        if i < len(files) - 1:
            time.sleep(10)

    # Write CSV
    with open(RESULTS_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "headline", "sentiment", "reason"])
        writer.writerows(results)

    print(f"\nResults saved to {RESULTS_CSV}")

    # Summary
    sentiments = [r[2] for r in results]
    print(f"\nSummary: {len(results)} pages")
    for s in ("positive", "negative", "neutral", "error"):
        count = sentiments.count(s)
        if count:
            print(f"  {s}: {count}")


if __name__ == "__main__":
    main()
