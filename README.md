# Income Predictor
A machine learning capstone project that predicts whether an individual's annual income exceeds $50,000, using demographic and employment data from the 1994 U.S. Census dataset. Built with a focus on fairness-aware data preparation and a comparison of an interpretable model against a higher-performing neural network.

# Problem
The label for this problem is income_binary (<=50K / >50K), used to aid community development organizations to identify individuals who may qualify for outreach programs targeting lower-income households. Because that use case makes false negatives (incorrectly predicting >50K for someone who actually qualifies) the more costly error, model selection throughout this project favors recall on the <=50K class, not just overall accuracy.

# Dataset
censusData.csv — 32,561 records with demographic and employment attributes (age, education, occupation, marital status, hours worked, capital gains/losses, etc.). The label is imbalanced: ~76% <=50K, ~24% >50K.

# Requirements
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow

# Usage
1. Place censusData.csv in a data/ folder in the project root.
2. Open Capstone.ipynb and run all cells in order.
3. The notebook will walk through EDA, data preparation, model training, hyperparameter tuning, and a side-by-side comparison of the decision tree and neural network.

# Future Work
- Complete a subgroup fairness evaluation: break down precision/recall by sex, race, and native country for both models
- Package the preprocessing and final model into a reusable pipeline, and build a simple interface for entering an individual's information and getting a prediction