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
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Conversion%20Rate%20by%20Group.png" alt="Conversion Rate" width="1000">
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
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Komposisi%20Group.png" alt="Komposisi Group" width="1000">
</p>

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Missing%20value.png"
    alt="Missing value" width="1000">
</p>

- A histogram of total ads reveals:
  - Where most users cluster in ad exposure (low vs high exposure).
  - How common extreme exposure is (very few vs very many ads).
  - Whether exposure is fairly tight or heavily skewed (long-tail users who see lots of ads).

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Distribution%20of%20Total%20ads.png" alt="Distribution Total ads" width="1000">
</p>

- The “conversion rate vs total ads (binned)” view highlights:
  - Whether conversion changes as exposure increases.
  - Whether the pattern differs between ad and psa.
  - Whether the effect looks steady or only appears at certain exposure ranges (e.g., only after many ads).

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Conversion%20rate%20vs%20total%20ads.png" alt="Conversion rate vs total ads" width="1000">
</p>

- Day/hour breakdown charts highlight:
  - Whether the gap between ad and psa is consistent across days and hours.
  - Which time slices show stronger or weaker conversion.
  - Why a single overall average can hide important swings.

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Conversion%20Rate%20by.png" alt="Conversion rate by" width="1000">
</p>

## How the Experiment Behaves (segments, tests, diagnostics, and planning)?

### What needs to be understood about the experiment’s behaviour?
- After computing the Ad vs PSA difference, check:
  - Whether the difference is consistent or “moves around” across slices (day, hour, exposure).
  - Whether the evidence is strong or could be noise.
  - Whether the result is big enough to care about, not just “statistically significant”.

### How to analyse that behaviour step by step?
- Establish the overall result first:
  - Calculate conversion rate for ad and psa.
  - Report the observed difference (Ad − PSA).
- Validate the difference using multiple checks (not one number):
  - Run a two-proportion z-test on conversion counts.
  - Run a chi-square test on the 2×2 conversion table.
  - Run a Welch t-test on the 0/1 conversion vectors (as an additional check).
  - Report a confidence interval for (CR_ad − CR_psa) using a Wald-style interval.
- Add an intuitive “is this luck?” check:
  - Run a permutation test by shuffling converted many times.
  - Compare the simulated differences to the observed difference.
  - Report the permutation p-value and show the null distribution histogram.
- Inspect simple segments to see where the result changes:
  - Plot conversion rate by most ads day for each group.
  - Plot conversion rate by most ads hour for each group.
  - Plot exposure behaviour:
    - Show the histogram of total ads (sampled).
    - Show conversion rate vs total ads bins for each group.
- Support planning for future tests:
  - Compute effect size using Cohen’s h from the observed conversion rates.
  - Estimate sample size per group for a chosen effect size and target power:
    - use statsmodels if available,
    - otherwise use the built-in fallback approximation.

### What does this reveal in practice?
- Reveal whether the Ad vs PSA gap looks stable or only appears in certain slices (day/hour/exposure).
- Provide “trust signals” beyond one p-value by combining:
  - z-test, chi-square, Welch t-test, confidence interval, and permutation simulation.
- Show whether the experiment is likely to hold up in real use, and how much sample size is needed to detect similar effects again.

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Difference%20%26%20CI.png" alt="Difference CI" width="1000">
</p>

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Permutation%20histogram.png" alt="Permutation Importance" width="1000">
</p>

# How reliable is the Ad vs PSA result?

## What does the final check say?
- Confirm the direction and size of the result using conversion rate in each group and the difference (Ad − PSA).
- Validate the same difference using more than one statistical check:
  - Two-proportion z-test (main test for proportions).
  - Chi-square test on the 2×2 conversion table.
  - Welch t-test on 0/1 outcomes (shown as an additional check in the app).
- Treat agreement across tests as a reliability signal.

<p align="center">
  <img src="https://github.com/Catherinerezi/Marketing-AB-Test-AD-vs-PSA/blob/main/assets/Covession%20Rate%20Table.png" alt="Conversion rate" width="1000">
</p>

## How often would this difference appear by luck?
- Simulate the “no real effect” world by shuffling converted values and re-splitting them into Ad-sized and PSA-sized groups.
- Compare the observed (Ad − PSA) difference against the simulated distribution.
- Read the permutation p-value as: how often random shuffles produce a difference at least this extreme.

## How does this connect back to earlier sections?
- Data checks ensure group comparison is not distorted by schema or missing-value issues.
- Segment views (day/hour/exposure) show where the gap strengthens or weakens.
- The reliability section adds the final trust layer: repeated tests + a shuffle-based check.

# What this Ad vs PSA A/B test really tells us?
## What is now known?
- **Ads may win, but the gap is small.** <br>
  Conversion is a rare event in this dataset, so the conversion rates look tiny. Even a small difference (Ad − PSA) can still matter for business decisions, but it should be read as “small uplift,” not a dramatic jump.

- **Marketing outcomes naturally look noisy.** <br>
  People see ads at different times and in different amounts (most ads hour, most ads day, total ads). Because of that, the conversion gap can appear stronger in some slices and weaker in others, even when the overall average is stable.

- **The dataset quality is strong, so the result is not driven by messy data.** <br>
  The dashboard shows missing-rate checks and basic schema validation (group labels and converted formatting). This keeps the comparison focused on behavior, not cleaning problems.

- **Trust comes from repeated checks, not from one number.** <br>
  The same Ad vs PSA difference is evaluated through multiple methods in the app (z-test, chi-square, plus the permutation check shown elsewhere). Agreement across checks supports confidence that the observed gap is not just luck, especially with the large sample size and imbalanced groups.

# What to do next?
### Use “uplift (Ad − PSA)” as the main A/B KPI
- Treat the primary KPI as conversion rate uplift: CR(ad) − CR(psa).
- Report it in two forms:
  - Absolute uplift (percentage points).
  - Relative lift (optional, for business framing).
- Track the KPI with the same definition across runs to avoid shifting conclusions.

### Require evidence, not just a higher conversion rate
- Avoid making a decision from uplift alone.
- Use at least one statistical confirmation already provided in the dashboard (z-test / chi-square) to support the claim that the gap is unlikely to be random.
- Treat “significant but tiny” as a different decision category from “significant and meaningful.”

### Check where the effect changes (simple slices)
- Review breakdown charts already present in the dashboard:
  - Conversion by most ads day.
  - Conversion by most ads hour.
  - Conversion vs total ads (binned).
- Use these slices to answer practical rollout questions:
  - Whether uplift appears consistently across days/hours.
  - Whether uplift grows only at higher total ads exposure.

### Plan the next experiment using power and sample size
- Use the power / sample-size section to estimate whether the current sample size is sufficient for a chosen minimum detectable effect (via Cohen’s h).
- Treat planning as mandatory when targeting small uplifts, because small effects require large samples.

### Turn the result into an action, not just a report
- If Ad wins with reliable evidence: consider rollout with monitoring, then re-check uplift in slices (day/hour/exposure).
- If no reliable difference appears: consider revising the ad creative, targeting, or exposure strategy, then rerun with a clearer expected effect size.
- If the effect only appears in certain slices: treat the result as a segmentation opportunity rather than a universal win.

### Standardize reporting for repeatable decisions
- Keep one consistent reporting set for every run:
  - CR(ad), CR(psa), uplift.
  - Significance check(s).
  - Slice views (day/hour/exposure).
  - Sample-size planning output.
- Preserve the same chart definitions to prevent “moving goalposts” between versions.
