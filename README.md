# 2025 MLDL Kaggle Competition – Predict Used Car Prices

Author: **Youngkee Kim**

This repository contains my solution for the **2025 MLDL Kaggle Competition – Predict Used Car Prices**, a regression competition evaluated by **MAPE (Mean Absolute Percentage Error)**.

In this competition, I achieved:
- **Public Leaderboard: 1st place out of 100+ participants**
- **Private Leaderboard: 2nd place out of 100+ participants**

---

## Evaluation Metric – MAPE

The official evaluation metric of this competition is **MAPE (Mean Absolute Percentage Error)**, defined as:

$$
\mathrm{MAPE} = \frac{100}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

where:
- \( y_i \) is the true price of car \( i \),
- \( \hat{y}_i \) is the predicted price of car \( i \),
- \( N \) is the total number of samples.

Lower MAPE means better performance, i.e., predictions are closer (in percentage terms) to the true used car prices.

---

## Files in This Repository

- `raw_data/train.csv` – Training data provided by the competition.
- `raw_data/test.csv` – Test data for which final predictions are generated.
- `raw_data/sample.csv` – Sample submission file from the competition.
- `model_log.ipynb` – Notebook used to experiment with models, log training, and track performance.
- `predict_new.ipynb` – Final inference notebook used to generate competition submissions.

The `catboost_info/` folder contains logs and metadata automatically generated during **CatBoost** model training (learning curves, metrics, etc.).

---

## What `predict_new.ipynb` Does

`predict_new.ipynb` is the **final prediction (inference) notebook** that prepares the test data, loads the trained model, and creates the submission file for Kaggle. The typical flow inside this notebook is:

1. **Import Libraries and Set Seed**
   - Import required packages (e.g., `pandas`, `numpy`, `catboost`).
   - Set a global random seed for reproducibility of any stochastic steps.

2. **Load Data**
   - Read `raw_data/test.csv` as the input test dataset.
   - Optionally load any additional files needed for preprocessing (e.g., encoders, feature lists, or configuration).

3. **Apply the Same Preprocessing as Training**
   The test data goes through exactly the same preprocessing pipeline as the training data so that the model receives features in the same format:
   - **Column selection & type casting**  
     - Keep only the columns actually used for modeling (drop ID-only or leak-prone columns if any).  
     - Cast each column to the intended type (e.g., `int`, `float`, `category`, `datetime`).
   - **Missing-value handling**  
     - For numerical features, fill missing values with the same rule used in training (e.g., median value or a sentinel like `-1`).  
     - For categorical features, fill missing values with a unified label such as `"Unknown"` or `"Missing"`.
   - **Feature engineering**  
     - Recreate derived features such as car age (e.g., `current_year - year`), mileage per year, or any interaction features used in the training notebook.  
     - Apply the same transformations (e.g., log transform for price-related variables, clipping of extreme mileage or age).
   - **Categorical encoding / ordering**  
     - Map categorical columns to the same categories and encodings as in training (e.g., label order or one-hot/target encoding).  
     - Ensure that category levels not seen in training are handled consistently (e.g., mapped to an `"Other"` category).
   - **Feature alignment**  
     - Reorder columns so that the final test feature matrix has the exact same column list and order as the training feature matrix.  
     - Drop any extra columns and add any missing columns with default values if needed.

4. **Load the Trained Model**
   - Load the final trained model (for example, a saved CatBoost model file from the training phase).
   - This model is usually trained and saved in a separate notebook such as `model_log.ipynb`.

5. **Generate Predictions**
   - Use the loaded model to predict used car prices for all rows in `test.csv`.
   - Store the predictions in a new column (e.g., `predicted_price`).

6. **Create Submission File**
   - Combine the prediction column with the required ID column from the test data.
   - Format the output to match the official Kaggle submission format (e.g., `id,price`).
   - Save the final CSV (e.g., `submission.csv`) that was uploaded to Kaggle.

This notebook is focused only on **inference and submission generation**, not on training. All model selection, hyperparameter tuning, and detailed experiments are handled in `model_log.ipynb` and related work, while `predict_new.ipynb` provides a clean, reproducible way to regenerate the final leaderboard submissions.

---

## Reproducibility

To reproduce the submission:
1. Place the competition data files in `raw_data/` (e.g., `train.csv`, `test.csv`, `sample.csv`).
2. Ensure the final trained model file (used by `predict_new.ipynb`) is available in the expected path.
3. Open `predict_new.ipynb` in Jupyter Notebook or JupyterLab.
4. Run all cells from top to bottom to generate the submission CSV.

This is the same process used to obtain the **Public 1st place** and **Private 2nd place** results in this competition.
