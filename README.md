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
  - Sample rows for heavy charts when needed to keep the dashboard responsive.

**Note:** Exclude duplicate-user validation from the current implementation, omit checks for repeated user id values unless explicitly added.

# Attachment
- [Data Processing](https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/raw-data/marketing_AB.csv)

# What This A/B Dashboard Brings to the Table

## Show usefulness through a “trust + action” view (not just one number)

### Answer the main question clearly
- Determine whether the Ad group converts more than the PSA group using converted (1 = converts, 0 = does not).
- Report the result as simple, decision-friendly outputs:
  - Conversion rate for each group.
  - The difference between groups (Ad − PSA).
  - A clear direction: which group performs better.
 
### Check whether the difference is likely real (not just luck)
- Avoid relying on a single calculation.
- Validate the group difference using:
  - Two-proportion z-test.
  - Chi-square test on the 2×2 conversion table.
  - Shuffle-based permutation test that simulates “no real effect” and compares against the observed difference.
- Provide a confidence interval for the conversion-rate difference to show a reasonable range for the uplift.

### Explain when the result changes (simple slices)
- Break down conversion results by the dataset context columns:
  - most ads day
  - most ads hour
  - total ads (binned)
- Support practical questions such as:
  - “Does the gap appear on every day or only certain days?”
  - “Does the gap change by hour?”
  - “Do heavier ad exposures behave differently?”
 
### Support experiment planning (power and sample size)
- Estimate approximate sample size needs based on a chosen effect size and target power.
- Provide a fallback calculation when optional libraries are unavailable, ensuring planning remains available.

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Conversion%20Rate%20by%20Group.png" alt="Visualisasi Perbandingan" width="1000">
</p>

## How Big the Problem Is (the shape of conversions & data quality)?

### What needs to be understood about the data?
- Recognize that marketing results can look “random” because people see ads at different times and in different amounts.
- Check whether the dataset is clean enough to trust before comparing groups.
- Confirm whether the Ad group and PSA group have enough data, and note that the group sizes are not balanced (Ad is much larger than PSA).

### How to explore this in practice (based on the dataset + Streamlit app)?
- Load marketing_AB.csv and confirm the table size (588,101 rows, 7 columns).
- Validate that key columns exist and are readable:
  - test group (expects ad and psa)
  - converted (boolean / 0–1 outcome)
  - total ads, most ads day, most ads hour, user id
- Verify basic data quality:
  - Check missing values per column (this dataset has 0% missing across columns).
  - Check duplicate rows (this dataset has 0 duplicates).
- Standardize values so group comparison stays fair:
  - Strip column-name spaces.
  - Normalize group labels to lowercase (ad, psa).
  - Convert converted into numeric 0/1 when needed (e.g., True/False → 1/0).
- Summarize the size and outcome of each group (this app shows it in the Overview/EDA sections):
  - Count users per group (Ad ≫ PSA).
  - Compute conversion rate per group from converted.
- Inspect “context columns” that can change results:
  - Break down conversion by most ads day.
  - Break down conversion by most ads hour.
  - Inspect total ads distribution (optionally sample rows for faster charts).
  - Compare conversion across total ads bins to see whether heavier exposure behaves differently.

### What picture of the problem appears after this check?
- Confirm that the dataset is clean (no missing, no duplicates), so differences are not driven by data errors.
- Observe that the experiment is heavily imbalanced in group size (many more rows in ad than psa), so interpretation should stay careful and rely on significance checks.
- See that conversion is a rare event overall (most rows are converted = 0), so small percentage differences can still matter.
- Notice that results can shift by day, hour, and ad exposure level, so “one average number” may hide where the effect is strongest or weakest.

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Komposisi%20Group.png" alt="Visualisasi Perbandingan" width="1000">
</p>

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Missing%20Value.png"
    alt="Before Data Cleaning" width="300">
</p>
