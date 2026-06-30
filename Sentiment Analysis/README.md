# Amazon Product Reviews — Sentiment Analysis

CodeAlpha Internship — Sentiment Analysis Task
**Author:** Nesma Yahia

## overview

Classifying real customer reviews as positive, negative, or neutral, detecting specific emotions, and turning the results into decisions a product or marketing team can act on.

## Goal

Determine whether a piece of text expresses a positive, negative, or neutral opinion — automatically, at scale. Instead of reading thousands of reviews one by one, the program labels every single one in seconds.

## What this project does

`amazon_sentiment_analysis.py` is a single, self-contained script that:

1. **Loads a real text dataset** — Amazon product reviews (review text + star rating + product name).
2. **Cleans the text** — strips URLs/HTML, fixes whitespace, repairs messy/duplicated product names; drops rows missing review text or a rating.
3. **Classifies sentiment** using [VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner) — a lexicon-based tool built for short, informal text like reviews and social media. Each review is labeled **Positive**, **Negative**, or **Neutral** from a compound polarity score.
4. **Detects specific emotions** — joy, trust, anger, sadness, fear, surprise — using a compact keyword lexicon (NRC-style).
5. **Validates the method** by comparing text sentiment against the actual star rating each reviewer gave.
6. **Visualizes the results** — five charts covering overall sentiment, sentiment share, emotion frequency, sentiment vs. star rating, and sentiment by product.
7. **Auto-generates a polished Word report** — `Amazon_Sentiment_Analysis_Report.docx` — explaining how the text was processed, how sentiment was assigned, the results, and conclusions for the business.

## Data source

This project uses the **Datafiniti Amazon Consumer Reviews of Amazon Products** dataset — ~34,660 real customer reviews of Amazon devices (Fire tablets, Kindle, Echo, Fire TV), including review text, star rating (1–5), and product name.

- Original source (Kaggle): https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
- Mirror used to download it directly:
  ```bash
  curl -L -o amazon_reviews.csv "https://raw.githubusercontent.com/Arjun-Mota/amazon-product-reviews-sentiment-analysis/master/1429_1.csv"
  ```

> Goodreads' `books.csv` was considered first (per the original task data), but it contains no review text — only numeric ratings — so it can't be used for text-based sentiment analysis. This dataset was chosen instead because it matches the task's "Amazon reviews" example exactly and includes both text and a ground-truth star rating to validate the model against.

### Using your own data

The script expects a CSV named `amazon_reviews.csv` in the same folder, with (at minimum) these columns:

| Expected column | Meaning |
|---|---|
| `reviews.text` | the review text |
| `reviews.rating` | star rating (1–5) |
| `name` | product name |

If your dataset uses different column names, update the `usecols` / `rename` step near the top of the script to match.

## Requirements

```bash
pip install pandas matplotlib seaborn numpy vaderSentiment python-docx
```

## Usage

1. Download `amazon_reviews.csv` (see Data source above) into the same folder as the script.
2. Run:

```bash
python3 amazon_sentiment_analysis.py
```

3. Outputs generated in the same folder:
   - `chart_1_sentiment_counts.png` … `chart_5_sentiment_by_product.png` — standalone chart images
   - `sentiment_summary.json` — structured summary stats
   - `Amazon_Sentiment_Analysis_Report.docx` — the full written report

## How sentiment is assigned

VADER scores each review from **-1** (most negative) to **+1** (most positive) using a compound score that accounts for negation ("not good"), intensity modifiers ("very good"), punctuation emphasis ("!!!"), and capitalization ("AMAZING"):

| Compound score | Label |
|---|---|
| ≥ 0.05 | Positive |
| -0.05 to 0.05 | Neutral |
| ≤ -0.05 | Negative |

## Key findings

- **90.2%** of reviews are positive, **5.5%** negative, **4.4%** neutral — public opinion is strongly favorable.
- Text sentiment agrees with the star rating **87.4%** of the time, validating the lexicon-based approach.
- **"Joy"** is the most frequently detected specific emotion (22,765 mentions), followed by trust.
- Sentiment varies meaningfully by product — useful for flagging individual SKUs that need a quality or support review, rather than relying on aggregate ratings alone.

## Project structure

```
.
├── amazon_reviews.csv                          # input dataset (not included — see Data source)
├── amazon_sentiment_analysis.py                 # main script (analysis + charts + report generation)
├── chart_1_sentiment_counts.png                 # generated
├── chart_2_sentiment_share.png                  # generated
├── chart_3_emotion_counts.png                   # generated
├── chart_4_sentiment_vs_rating.png              # generated
├── chart_5_sentiment_by_product.png             # generated
├── sentiment_summary.json                       # generated
└── Amazon_Sentiment_Analysis_Report.docx        # generated
```

## 🌐 Connect with Me

- 💼 [LinkedIn](https://www.linkedin.com/in/nesma-yahia-801a92237/)
- 🐦 [X (Twitter)](https://x.com/yahianesma71)
- 💻 [GitHub](https://github.com/Nesma07)
- 
