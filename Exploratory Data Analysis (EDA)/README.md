# Goodreads Books — Exploratory Data Analysis

CodeAlpha Internship — Task 2
**Author:** Nesma Yahia

## Overview

This project performs an end-to-end exploratory data analysis (EDA) on the Goodreads Books dataset. Running a single script loads the data, cleans it, analyzes it, and automatically produces both a visual dashboard and a polished Word report — no manual steps in between.

## What the script does

1. **Load & inspect** — reads `books.csv`, prints shape, dtypes, and missing-value counts.
2. **Handle missing data** — fills missing categorical fields (`publisher`, `language_code`, `authors`, `title`) with `'Unknown'`, and drops rows missing essential numeric fields (rating, page count, ratings count, etc.), since those can't be safely imputed.
3. **Clean** — removes invalid records (0-page or 0-rating books), parses publication year, and log-transforms ratings count for plotting.
4. **Univariate analysis** — descriptive stats on rating, page count, ratings count, plus top authors, top publishers, and language distribution.
5. **Correlation analysis** — Pearson correlations between rating, pages, ratings count, and review count.
6. **Visual dashboard** — an 11-chart dashboard (`goodreads_eda.png`) covering rating distribution, top authors/publishers, language mix, page-count distribution, ratings-vs-rating scatter, publication trend over time, average rating by language, correlation heatmap, average rating by page range, and the top 10 most-rated books.
7. **Key findings & plain-English observations** — printed to console.
8. **Word report export** — generates `Goodreads_EDA_Report.docx`, a formatted report containing the dataset description, missing-data methodology, key statistics, correlation matrix, embedded dashboard image, findings, and conclusion. All values are computed live from the data, not hardcoded.

## Files

| File | Description |
|---|---|
| `Task_eda.py` | Main script — run this. Produces the PNG and DOCX. |
| `books.csv` | Input dataset (not included — see below). |
| `goodreads_eda.png` | Generated dashboard (11 charts). |
| `Goodreads_EDA_Report.docx` | Generated Word report. |

## Requirements

```bash
pip install pandas numpy matplotlib seaborn python-docx
```

## Usage

1. Place a `books.csv` file in the same folder as `Task_eda.py`. Expected columns:
   `bookID, title, authors, average_rating, isbn, isbn13, language_code, num_pages, ratings_count, text_reviews_count, publication_date, publisher`
2. Run the script:

```bash
python Task_eda.py
```

3. Output appears in the same folder:
   - `goodreads_eda.png`
   - `Goodreads_EDA_Report.docx`

## Dataset

The [Goodreads Books dataset](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) contains metadata for ~11,000 books scraped from Goodreads, including ratings, page counts, authors, publishers, and language.

## Key Findings (example run)

- Average rating across the dataset: **3.94 / 5** (std. dev. 0.30)
- Median page count: **302 pages**
- Most prolific author: **Stephen King** (40 books)
- English-language books make up **~93%** of the dataset
- Ratings count and review count are strongly correlated (**r ≈ 0.87**)
- Page count has only a weak relationship with average rating (**r ≈ 0.17**)
- Most-rated book: **Twilight** (~4.6M ratings)

*(Exact figures depend on the specific `books.csv` used and are recalculated automatically on each run.)*
