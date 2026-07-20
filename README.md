# CarPricePredictor

A machine learning project that predicts used car prices in India based on features like make, year, kilometers driven, fuel type, engine specs, and physical dimensions.

## Project Overview

This project takes a raw used-car listings dataset, cleans and engineers features from it, trains several regression models, and identifies the best-performing model for predicting price. It's built as a learning project covering the full pipeline from raw data to a tuned model, with containerized deployment as the next milestone.

## Dataset

The raw dataset includes listings with columns such as Make, Model, Price, Year, Kilometer, Fuel Type, Transmission, Location, Color, Owner, Seller Type, Engine, Max Power, Max Torque, Drivetrain, Length, Width, Height, Seating Capacity, and Fuel Tank Capacity.

## What's Been Done

### Data Cleaning
- Identified rows with missing values, which clustered around specific makes/models (Engine, Max Power, Max Torque, Drivetrain, Length, and Width were often missing together, suggesting a different data source for those rows).
- Dropped rows with missing values after confirming the row loss was acceptable (~1874 rows remaining).
- Dropped the `Model` column (Make was considered sufficiently descriptive) and `Engine` (redundant with Max Power/Max Torque).

### Feature Engineering
- **Volume**: Combined `Length`, `Width`, and `Height` into a single `Volume` feature (converted to cubic meters) instead of keeping three separate dimension columns.
- **Power/Torque split**: The original `Max Power` and `Max Torque` columns were strings like `"87 bhp @ 6000 rpm"`. These were split into four numeric columns using regex extraction: `Power_bhp`, `Power_rpm`, `Torque_Nm`, and `Torque_rpm`. Rows where extraction failed (formatting inconsistencies in the source data) were dropped.

### Preprocessing
- Built a `ColumnTransformer` combining:
  - `StandardScaler` for numeric features
  - `OneHotEncoder` (with `handle_unknown='ignore'`) for categorical features (Make, Fuel Type, Transmission, Location, Color, Owner, Seller Type, Drivetrain)
- Data was split into train/test sets **before** fitting the preprocessor, to avoid data leakage. The preprocessor is fit only on training data and reused (via `.transform()`) on the test set.

### Model Training & Comparison
Trained and compared the following regressors on Test R2:

| Model | Test R2 |
|---|---|
| Random Forest Regressor | 0.814 |
| K-Neighbors Regressor | 0.695 |
| Lasso | 0.684 |
| Linear Regression | 0.683 |
| Decision Tree | 0.682 |
| Ridge | 0.679 |
| AdaBoost Regressor | 0.354 |

**Random Forest Regressor** was the strongest performer, though it showed a noticeable train/test R2 gap (0.980 train vs. 0.814 test), indicating some overfitting.

### Hyperparameter Tuning
Used `RandomizedSearchCV` (5-fold CV, scoring on R2) to tune the Random Forest across `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `bootstrap`, aiming to narrow the overfitting gap while maintaining or improving test performance.

### Model Persistence
The tuned model and the fitted preprocessor are saved separately with `joblib` so that new raw input can be transformed consistently at inference time:
- `models/random_forest_model.pkl`
- `models/preprocessor.pkl`

## Project Structure

```
CarPricePredictor/
├── app/                  # Serving/inference code (in progress)
├── data/carcsvdata/      # Raw and cleaned datasets
├── models/               # Saved model and preprocessor (.pkl)
├── notebooks/            # exploration.ipynb, modeltraining.ipynb
├── src/                  # Reusable pipeline modules (in progress)
├── tests/                # Unit tests (in progress)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/amitojphagura/CarPricePredictor.git
cd CarPricePredictor
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

Open the `notebooks/` folder in VS Code or Jupyter to explore the analysis and training process.

## Future Work

This project is not yet complete. Remaining steps:

- [ ] **Address the Random Forest train/test overfitting gap** — evaluate whether the tuned hyperparameters from `RandomizedSearchCV` sufficiently close the gap, or whether further constraints (e.g. lower `max_depth`, higher `min_samples_leaf`) are needed.
- [ ] **Residual analysis** — plot predicted vs. actual and residuals vs. predicted to check for systematic errors (e.g. underperformance on luxury/rare cars).
- [ ] **Test unseen-category handling** — confirm the pipeline behaves sensibly on genuinely new categorical values not seen during training.
- [ ] **Refactor notebooks into `.py` scripts** — move data loading, preprocessing, training, and evaluation logic out of Jupyter notebooks and into modular scripts under `src/` (e.g. `data.py`, `model.py`, `train.py`, `predict.py`).
- [ ] **Build an inference interface** — decide on and implement a serving method (e.g. a FastAPI or Flask app under `app/serve.py`) that loads the saved model and preprocessor and returns predictions for new input.
- [ ] **Write unit tests** — add tests under `tests/` to confirm the loaded model and preprocessor produce consistent, expected predictions on fixed sample inputs.
- [ ] **Containerize the application** — write a `Dockerfile` and, if needed, a `docker-compose.yml` to package the app, model, and dependencies into a deployable container.
- [ ] **Finalize `requirements.txt`** — ensure it reflects an exact, pinned set of dependencies (via `pip freeze`) matching what the saved model was trained and tested with.
- [ ] **Documentation polish** — add a project description and topics to the GitHub repo's About section, and expand this README with example predictions once the inference interface is built.
