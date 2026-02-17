# Estimating Incremental Uplift with Confidence: An Explainable A/B Testing System (Ad vs PSA)

Marketing results can change a lot: people come on different days and hours, some people see the message many times (total ads) while others see it only a little, and most people simply do not convert. In your raw file (marketing_AB.csv), the Ad group is much bigger than the PSA group, and conversions are rare, so it’s easy to get confused by numbers if we don’t check carefully. But we still want one dependable answer: does an Ad really make more people convert than a PSA?

The goal of this project is to use the real dataset to compare the Ad message vs the PSA message on the outcome converted, and make the conclusion clear enough to support a decision (not just a guess).

The project focuses on these core tasks:
- Make the data safe to use: load the CSV, confirm the needed columns exist, clean group labels (ad/psa), and make sure converted is read correctly as True/False (1/0).
- Compare the two groups clearly: calculate the conversion rate for ad and psa, and show the difference.
- Check if the difference is real or could be luck: run simple math checks (the tests in the app) and a shuffle simulation (permutation test).
- Show when the result changes: break the comparison into easy slices using most ads day, most ads hour, and total ads, and include a planning section (power/sample size) so we know if the test is big enough.

The Streamlit app puts all of this into one place—data checks, the main Ad vs PSA comparison, simple charts by day/hour/exposure, reliability checks, and experiment planning, so someone can understand what happened, how sure we should be, and what to do next (keep the ad, improve it, target it, or test longer).

**Two groups explanation:**<br>
In this project, we compare two groups of people:
- Ad group: people who see an advertisement (test group = ad)
- PSA group: people who see a normal message, not an ad (test group = psa)<br>
Then we look at converted:
- converted = 1 (True) means the person buys/signs up
- converted = 0 (False) means the person does not

# Understand the Experiment Data (Ad vs PSA)

Treat each row as one person in the test:
- Identify who appears → user id
- Identify what was shown → test group (ad or psa)
- Identify whether conversion happened → converted (yes/no)
- Measure how many ads were seen → total ads
- Locate when exposure peaked → most ads day, most ads hour
Ensure clean, readable columns before comparing results to prevent misleading conclusions.

| Column          | Type (in file)      | Meaning                    | Notes & Handling in the app                                                           |
| --------------- | ------------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| `Unnamed: 0`    | `int`               | Row index                  | Treat as a saved index column; exclude from analysis.                                 |
| `user id`       | `int`               | Person identifier          | Keep for reference; avoid using as a predictive signal (A/B comparison only).         |
| `test group`    | `text`              | Assigned message group     | Require both `ad` and `psa`; standardize case and spacing to avoid label errors.      |
| `converted`     | `bool` (True/False) | Conversion outcome         | Convert safely into numeric **0/1** for consistent calculations.                      |
| `total ads`     | `int`               | Number of ads seen         | Expect wide range (up to **2065** in this file); use binning and sampling for charts. |
| `most ads day`  | `text`              | Day with highest exposure  | Use for day-level conversion comparisons (Ad vs PSA).                                 |
| `most ads hour` | `int` (0–23)        | Hour with highest exposure | Use for hour-level conversion comparisons (Ad vs PSA).                                |

**Apply quality checks and cleaning rules**<br>
Run A/B comparison, not machine learning, prioritize a clean and fair group comparison.
- Standardize column names
  - Strip extra spaces from column names to ensure consistent access.
- Normalize converted
  - Convert boolean True/False into 1/0.
  - Convert text-like values such as "true"/"false" or "1"/"0" into numeric form when present.
- Normalize test group
  - Trim spaces and convert labels to lowercase.
  - Stop execution if both ad and psa are not present.
- Report basic data health
  - Display dataset shape, data types, and missing percentages.
  - Display group sizes (row counts for ad and psa).
- Control chart performance
  - Sample rows for heavy charts when needed to keep the dashboard responsive. <br>
**Note:** Exclude duplicate-user validation from the current implementation; omit checks for repeated user id values unless explicitly added.
