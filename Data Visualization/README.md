# Goodreads Books — Data Visualization

CodeAlpha Internship — Data Visualization Task
**Author:** Nesma Yahia

## Overview

Turning raw book metadata into clear, decision-supporting charts and a written data story.

## Goal

Transform numbers into visuals that are understood instantly — instead of saying "ratings and reviews are correlated," show a heatmap and the relationship is obvious at a glance.

## What this project does

`goodreads_data_visualization.py` is a single, self-contained script that:

1. **Loads and cleans the data** — handles missing values (fills non-essential text fields, drops rows missing core numeric fields) and removes invalid placeholder rows (0 pages, 0 rating).
2. **Builds five chart types**, each chosen for what it communicates best:

   | Chart | File | Used for |
   |---|---|---|
   | Bar chart | `chart_1_bar_top_authors.png` | Comparing categories — most prolific authors |
   | Line chart | `chart_2_line_publications_over_time.png` | Trends over time — books published per year |
   | Pie chart | `chart_3_pie_language_share.png` | Proportions, used sparingly — language mix |
   | Histogram | `chart_4_histogram_rating_distribution.png` | Distribution of values — average rating spread |
   | Heatmap | `chart_5_heatmap_correlations.png` | Correlations between variables |

3. **Tells the story** — each chart comes with a short, plain-English narrative explaining what it shows and why it matters (printed to console, saved to `data_story.json`, and embedded in the final report).
4. **Auto-generates a polished Word report** — `Goodreads_Data_Visualization_Report.docx`, with all five charts, their narratives, and a closing section connecting the findings to a real decision-making takeaway.

## Requirements

```bash
pip install pandas matplotlib seaborn numpy python-docx
```

## Usage

1. Place `books.csv` (Goodreads Books dataset) in the same folder as the script.
2. Run:

```bash
python3 goodreads_data_visualization.py
```

3. Outputs are generated in the same folder:
   - `chart_1_bar_top_authors.png` … `chart_5_heatmap_correlations.png` — standalone chart images (portfolio-ready)
   - `data_story.json` — structured summary stats + narrative text per chart
   - `Goodreads_Data_Visualization_Report.docx` — the full written report

## Dataset

[Goodreads Books dataset](https://github.com/Yannawut/Goodreads_Analysis) — ~11,000 books with title, author(s), average rating, page count, ratings count, text reviews count, publication date, publisher, and language code.

## Key findings

- English-language books make up ~80% of the catalogue.
- Ratings count and text review count are strongly correlated (0.87) — engaged readers do both.
- Page count has almost no relationship to average rating (0.17) — longer books aren't rated better or worse.
- Publishing volume in the dataset peaks around 2006, most likely reflecting when the dataset was scraped rather than an actual industry trend.
- Average ratings cluster tightly around 3.9–4.0, showing a fairly symmetric, slightly positive-skewed distribution.

## Project structure

```
.
├── books.csv                                    # input dataset (not included — add your own)
├── goodreads_data_visualization.py               # main script (charts + report generation)
├── chart_1_bar_top_authors.png                   # generated
├── chart_2_line_publications_over_time.png        # generated
├── chart_3_pie_language_share.png                 # generated
├── chart_4_histogram_rating_distribution.png      # generated
├── chart_5_heatmap_correlations.png               # generated
├── data_story.json                                # generated
└── Goodreads_Data_Visualization_Report.docx       # generated
```


## 🌐 Connect with Me

- 💼 [LinkedIn](https://www.linkedin.com/in/nesma-yahia-801a92237/)
- 🐦 [X (Twitter)](https://x.com/yahianesma71)
- 💻 [GitHub](https://github.com/Nesma07)
