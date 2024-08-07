# Causal Tasks with DoWhy

There are a number of causal questions that we could ask about a given dataset. Some of these include:

1. Effect estimation
   - If we change X, how much will it cause Y to change?
2. Attribution
   - Why did an event happen?
   - How to explain an outcome?
   - Which of my variables caused the anomaly?
3. Counterfactual estimation
   - What if X had been changed to a different value than its observed one?
4. Prediction
   - Given a new input with different input features, what will be the output?

# DoWhy-attribution

This repository contains the source code and the data pertaining to the *attribution* task. In particular, it focuses on explaining an unusually high value for the *p38* protein expression measurement. 

You can find the Python notebook, *outliers.ipynb*, which will have the code and methodology for how I was able to detect outliers in the dataset. At its core, I was able to utilize sci-kit learn's Isolation Forest and set the contamination to 0.0005. The reason for such a low contamination value is so I can filter out the dataset with high confidence that the outliers outputted by the algorithm really are data instances worth analyzing.

# Sachs Dataset
The Sachs dataset contains quantitative measurements about the expression levels of 11 phosphorylated protein and phospholipids in the human primary T cell. This dataset contains continuous & numeric data which is formatted as a txt tab spaced file.




