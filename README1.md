# Employee Attrition Prediction

A machine learning project that predicts which employees are likely to leave a company, so HR can take early action to retain them — instead of reacting only after resignation.

## Problem Statement

Companies often lose valuable employees unexpectedly. Replacing an employee is costly and time-consuming (recruiting, onboarding, training) and hurts team productivity. HR teams usually only find out an employee is unhappy after they've already resigned — too late to act.

## Solution

This project builds a classification model that predicts, in advance, which employees are at higher risk of leaving, based on patterns in their data such as overtime, income, age, and tenure. HR can use this to identify at-risk employees early and intervene — adjusting workload, revisiting pay, or improving engagement.

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets) (Kaggle) — structured employee records with fields like Age, Department, MonthlyIncome, OverTime, JobRole, and the target column `Attrition` (Yes/No).

## Project Workflow

1. **Exploratory Data Analysis (EDA)** — 5 charts examining attrition by department, overtime, income, age, and a correlation heatmap
2. **Encoding** — converted categorical text columns into numeric form
3. **Train/Test Split** — 80/20 split
4. **Initial Model** — trained a Random Forest Classifier
5. **Problem Found** — high accuracy (86.7%) but very low recall (5.1%) for employees who actually left, due to class imbalance in the dataset
6. **Fix #1 — Class Weighting** — retrained with `class_weight='balanced'`, improving recall to 10.3%
7. **Fix #2 — Lower Decision Threshold** — lowered the prediction confidence threshold from 50% to 30%, improving recall further to 28.2%
8. **Feature Importance** — identified the top factors driving attrition

## Results

| Stage | Accuracy | Recall (Attrition = Yes) |
|---|---|---|
| Initial model | 86.7% | 5.1% |
| After class balancing | 87.4% | 10.3% |
| After balancing + lower threshold | 85.4% | **28.2%** |

**Top factors driving attrition:** OverTime, MonthlyIncome, Age, DailyRate

## Key Insight

The initial model looked accurate but was practically useless — it barely caught any real attrition cases. Diagnosing this as a class imbalance problem and addressing it (class weighting + threshold tuning) meaningfully improved the model's ability to catch at-risk employees, at a small, reasonable cost to overall accuracy.

## Tools Used

Python, Pandas, Seaborn, Matplotlib, Scikit-learn (Random Forest Classifier)

## How to Run

1. Download the dataset CSV and place it in the project folder
2. Open `employee_attrition_prediction.ipynb` in Jupyter Notebook
3. Run all cells in order
