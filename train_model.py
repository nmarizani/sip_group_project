import argparse
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
 
RANDOM_STATE = 42
FEATURES = ['Crop', 'Season_Mapped', 'Area', 'Annual_Rainfall']
TARGETS = ['Yield', 'Fertilizer_per_area', 'Pesticide_per_area']
CAT_COLS = ['Crop', 'Season_Mapped']
 
SEASON_MAP = {
    'Kharif': 'Rainy Season', 'Autumn': 'Rainy Season',
    'Rabi': 'Dry Season', 'Winter': 'Dry Season', 'Summer': 'Dry Season',
    'Whole Year': 'Year-Round',
}
 
SCRIPT_DIR = Path(__file__).resolve().parent
 
 
def find_data_file(explicit_path):
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"--data path given but not found: {p}")
    # Look in this script's own folder, then one level up -- covers the common
    # cases of "csv next to the script" or "csv in the project root, script in api/"
    candidates = [
        SCRIPT_DIR / 'crop_yield.csv',
        SCRIPT_DIR.parent / 'crop_yield.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find crop_yield.csv. Put it in the same folder as this script, "
        "or run with: python train_model.py --data path\\to\\crop_yield.csv"
    )
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, help='Path to crop_yield.csv')
    parser.add_argument('--out', default=None, help='Output folder for the saved model (defaults to this script\'s folder)')
    args = parser.parse_args()
 
    data_path = find_data_file(args.data)
    out_dir = Path(args.out) if args.out else SCRIPT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
 
    print(f'Reading data from: {data_path}')
    print(f'Saving artifact to: {out_dir}')
 
    # ---- Reproduce cleaning (same steps as the notebook) ----
    df = pd.read_csv(data_path)
    df['Crop'] = df['Crop'].str.strip()
    df['Season'] = df['Season'].str.strip()
    df = df.drop(columns=['State'])
    df['Season_Mapped'] = df['Season'].map(SEASON_MAP)
    df['Fertilizer_per_area'] = df['Fertilizer'] / df['Area']
    df['Pesticide_per_area'] = df['Pesticide'] / df['Area']
 
    def cap_within_crop(frame, col, lo=0.01, hi=0.99):
        lower = frame.groupby('Crop')[col].transform(lambda s: s.quantile(lo))
        upper = frame.groupby('Crop')[col].transform(lambda s: s.quantile(hi))
        return frame[(frame[col] >= lower) & (frame[col] <= upper)]
 
    for col in ['Yield', 'Fertilizer_per_area', 'Pesticide_per_area']:
        df = cap_within_crop(df, col)
 
    X, y = df[FEATURES], df[TARGETS]
 
    # ---- Winning hyperparameters from the notebook's tuning (section 6b) ----
    gb_estimator = GradientBoostingRegressor(
        n_estimators=300, n_iter_no_change=10, validation_fraction=0.1, tol=1e-4,
        random_state=RANDOM_STATE,
        learning_rate=0.1, max_depth=4, subsample=1.0, min_samples_leaf=1,
    )
 
    pipeline = Pipeline([
        ('prep', ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
        ], remainder='passthrough')),
        ('model', MultiOutputRegressor(gb_estimator))
    ])
 
    pipeline.fit(X, y)
 
    # ---- Save artifact + metadata needed by the API for input validation ----
    joblib.dump(pipeline, out_dir / 'yield_model.joblib')
 
    metadata = {
        'features': FEATURES,
        'targets': TARGETS,
        'valid_crops': sorted(df['Crop'].unique().tolist()),
        'valid_seasons': sorted(df['Season_Mapped'].unique().tolist()),
        'area_range_seen': [float(df['Area'].min()), float(df['Area'].max())],
        'rainfall_range_seen': [float(df['Annual_Rainfall'].min()), float(df['Annual_Rainfall'].max())],
    }
    with open(out_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
 
    print('Trained on', len(df), 'rows.')
    print('Artifact saved:', out_dir / 'yield_model.joblib')
    print('Metadata saved:', out_dir / 'model_metadata.json')
    print()
    print('Sample sanity-check predictions:')
    sample = pd.DataFrame([
        {'Crop': 'Rice', 'Season_Mapped': 'Rainy Season', 'Area': 50, 'Annual_Rainfall': 1200},
        {'Crop': 'Wheat', 'Season_Mapped': 'Dry Season', 'Area': 20, 'Annual_Rainfall': 800},
    ])
    preds = pipeline.predict(sample)
    for i, row in sample.iterrows():
        print(dict(row), '->', dict(zip(TARGETS, preds[i].round(3))))
 
 
if __name__ == '__main__':
    main()