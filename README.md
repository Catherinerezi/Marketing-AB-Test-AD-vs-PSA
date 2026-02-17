# Estimating Incremental Uplift with Confidence: An Explainable A/B Testing System (Ad vs PSA)

Marketing results can change a lot: people come on different days and hours, some people see the message many times (total ads) while others see it only a little, and most people simply do not convert. In your raw file (marketing_AB.csv), the Ad group is much bigger than the PSA group, and conversions are rare, so it’s easy to get confused by numbers if we don’t check carefully. But we still want one dependable answer: does an Ad really make more people convert than a PSA?

The goal of this project is to use the real dataset to compare the Ad message vs the PSA message on the outcome converted, and make the conclusion clear enough to support a decision (not just a guess).

The project focuses on these core tasks:
- Make the data safe to use: load the CSV, confirm the needed columns exist, clean group labels (ad/psa), and make sure converted is read correctly as True/False (1/0).
- Compare the two groups clearly: calculate the conversion rate for ad and psa, and show the difference.
- Check if the difference is real or could be luck: run simple math checks (the tests in the app) and a shuffle simulation (permutation test).
- Show when the result changes: break the comparison into easy slices using most ads day, most ads hour, and total ads, and include a planning section (power/sample size) so we know if the test is big enough.

The Streamlit app puts all of this into one place—data checks, the main Ad vs PSA comparison, simple charts by day/hour/exposure, reliability checks, and experiment planning—so someone can understand what happened, how sure we should be, and what to do next (keep the ad, improve it, target it, or test longer).

**Two groups explanation:**
In this project, we compare two groups of people:
- Ad group: people who see an advertisement (test group = ad)
- PSA group: people who see a normal message, not an ad (test group = psa)
Then we look at converted:
- converted = 1 (True) means the person buys/signs up
- converted = 0 (False) means the person does not
