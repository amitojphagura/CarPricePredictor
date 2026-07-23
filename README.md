# CarPricePredictor

A machine learning project that predicts used car prices in India based on features like make, year, kilometers driven, fuel type, engine specs, and physical dimensions.

## Project Overview

This project takes a raw used-car listings dataset, cleans and engineers features from it, trains several regression models, identifies the best-performing model, and serves it as a containerized prediction API. The exploratory work started in Jupyter notebooks and has since been ported into a full `src/` pipeline (ingestion → transformation → training → prediction), wrapped in a Flask API, and packaged into a Docker image, anyone can pull this repo and run it without installing Python or any dependencies locally.

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
The tuned model and the fitted preprocessor are saved separately with `pickle` so that new raw input can be transformed consistently at inference time:
- `models/model.pkl`
- `models/preprocessor.pkl`

(`models/random_forest_model.pkl` is an earlier artifact saved directly from the notebook via `joblib` — superseded by `models/model.pkl`, kept around as a reference.)

### Pipeline Scripts (`src/`)
The notebook logic has been ported into a reusable pipeline:

- **`src/components/data_ingestion.py`** — reads the raw CSV, applies the cleaning/feature-engineering steps from `exploration.ipynb` (drop `Model`/`Engine`, build `Volume`, parse `Power_bhp`/`Power_rpm`/`Torque_Nm`/`Torque_rpm` out of the free-text power/torque columns), and splits into train/test sets under `artifacts/`.
- **`src/components/data_transformation.py`** — builds the same `OneHotEncoder` + `StandardScaler` `ColumnTransformer` from `modeltraining.ipynb` and saves it to `models/preprocessor.pkl`.
- **`src/components/model_trainer.py`** — compares Linear/Lasso/Ridge/KNN/DecisionTree/RandomForest/AdaBoost, then runs a `RandomizedSearchCV` tuning pass on Random Forest (mirroring the notebook), saving the winner to `models/model.pkl`.
- **`src/pipeline/train_pipeline.py`** — chains all three steps into one command (see "Retraining" below).
- **`src/pipeline/predict_pipeline.py`** — `CustomData` (wraps a single car's raw fields into the right dataframe shape) and `PredictPipeline` (loads the saved model/preprocessor and returns a prediction).
- **`src/exception.py`, `src/logger.py`, `src/utils.py`** — shared error handling, logging, and save/load helpers used throughout.

### Serving API (`app/`)
**`app/main.py`** is a small Flask app exposing:
- `GET /` — health check, lists the required prediction fields.
- `POST /predict` — accepts a JSON body with the fields below and returns `{"predicted_price": ...}`.

Required fields: `make`, `year`, `kilometer`, `fuel_type`, `transmission`, `location`, `color`, `owner`, `seller_type`, `drivetrain`, `seating_capacity`, `fuel_tank_capacity`, `volume`, `power_bhp`, `power_rpm`, `torque_nm`, `torque_rpm`.

(`app.py` at the project root is just a `python app.py` convenience shim that runs the same app — the real implementation lives in `app/main.py` since a top-level `app.py` and an `app/` package can't coexist under the same import name.)

### Containerization
A `Dockerfile` packages the app, the pipeline, the saved model, and all dependencies into a single image built on `python:3.12-slim`, served via `gunicorn`. No local Python install is required to run it — just Docker.

## Project Structure

```
CarPricePredictor/
├── app/
│   ├── main.py           # Flask app (GET /, POST /predict)
│   └── __init__.py
├── app.py                 # convenience entrypoint: `python app.py`
├── data/carcsvdata/        # raw and cleaned datasets
├── models/                 # saved model.pkl + preprocessor.pkl
├── notebooks/               # exploration.ipynb, modeltraining.ipynb (original analysis)
├── src/
│   ├── components/          # data_ingestion.py, data_transformation.py, model_trainer.py
│   ├── pipeline/             # train_pipeline.py, predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── tests/                   # unit tests (not yet written)
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Setup (running locally, without Docker)

```bash
git clone https://github.com/amitojphagura/CarPricePredictor.git
cd CarPricePredictor
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Making Predictions

There are three ways to get a prediction out of this project, from quickest to most portable.

### 1. Directly in Python (no server)

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

data = CustomData(
    make="Honda", year=2017, kilometer=87150, fuel_type="Petrol",
    transmission="Manual", location="Pune", color="Grey", owner="First",
    seller_type="Corporate", drivetrain="FWD", seating_capacity=5.0,
    fuel_tank_capacity=35.0, volume=10088.32, power_bhp=87.0,
    power_rpm=6000.0, torque_nm=109.0, torque_rpm=4500.0,
)
prediction = PredictPipeline().predict(data.get_data_as_data_frame())
print(prediction[0])
```

Run from the project root (with the venv activated) so the relative `models/` paths resolve correctly.

### 2. Local Flask server

```bash
python app.py
```

Then, from another terminal:

```bash
curl http://localhost:5000/
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"make\": \"Honda\", \"year\": 2017, \"kilometer\": 87150, \"fuel_type\": \"Petrol\", \"transmission\": \"Manual\", \"location\": \"Pune\", \"color\": \"Grey\", \"owner\": \"First\", \"seller_type\": \"Corporate\", \"drivetrain\": \"FWD\", \"seating_capacity\": 5.0, \"fuel_tank_capacity\": 35.0, \"volume\": 10088.32, \"power_bhp\": 87.0, \"power_rpm\": 6000.0, \"torque_nm\": 109.0, \"torque_rpm\": 4500.0}"
```

### 3. Docker (no Python install needed at all)

```bash
docker build -t car-price-api .
docker run -p 5000:5000 car-price-api
```

Then hit it exactly the same way as option 2, via `curl` (or a browser for `GET /`) at `http://localhost:5000`.

## Retraining the Model

To rerun the full ingestion → transformation → training pipeline (e.g. after changing the data or the model grids):

```bash
python -m src.pipeline.train_pipeline
```

This regenerates `models/model.pkl` and `models/preprocessor.pkl`, and prints the final test R² score. Note this runs the full multi-model comparison plus a `RandomizedSearchCV` tuning pass, so it takes a few minutes rather than seconds.

## Future Work

The core pipeline, API, and containerization are done. What's left:

- [ ] **Local web frontend** — a simple page (served either from Flask itself via a `templates/` folder, or a separate small frontend) where you can fill in a form with a car's details and get a predicted price back, running on `localhost`, instead of needing `curl`/Postman.
- [ ] **Address the Random Forest train/test overfitting gap** — evaluate whether the tuned hyperparameters sufficiently close the gap (0.980 train vs. 0.814 test in the original notebook run), or whether further constraints (e.g. lower `max_depth`, higher `min_samples_leaf`) are needed.
- [ ] **Write unit tests** — add tests under `tests/` to confirm the loaded model and preprocessor produce consistent, expected predictions on fixed sample inputs.
