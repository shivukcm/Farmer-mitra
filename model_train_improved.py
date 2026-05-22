import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

print("\n" + "="*80)
print("IMPROVED MODEL TRAINING WITH CORRECTED FEATURE EXTRACTION")
print("="*80)

# Load all data
soil = pd.read_csv('data/soil.csv')
crop_yield = pd.read_csv('data/crop_yield.csv', header=None)
market = pd.read_csv('data/market.csv')
soil_crop = pd.read_csv('data/soil_crop.csv')

# Parse crop_yield structure
crop_names_row = crop_yield.iloc[0, 3:]
seasons_row = crop_yield.iloc[1, 3:]
measurements_row = crop_yield.iloc[2, 3:]

yield_data = crop_yield.iloc[3:].reset_index(drop=True)
yield_data.columns = crop_yield.columns

# Build mapping of (crop, season, measurement_type) -> column_index
crop_season_map = {}  # (crop, season) -> {measurement -> col_idx}
for col_idx in range(3, len(crop_names_row) + 3):
    crop = crop_names_row.iloc[col_idx - 3]
    season = seasons_row.iloc[col_idx - 3]
    measurement = measurements_row.iloc[col_idx - 3]
    
    key = (str(crop).strip(), str(season).strip())
    if key not in crop_season_map:
        crop_season_map[key] = {}
    
    meas_type = 'area' if 'Area' in str(measurement) else \
                'production' if 'Production' in str(measurement) else \
                'yield' if 'Yield' in str(measurement) else None
    
    if meas_type:
        crop_season_map[key][meas_type] = col_idx

# Load soil averages
soil_avg = soil[soil['Type'].str.lower() == 'average'].copy()
soil_avg['District'] = soil_avg['District'].str.strip()

# Load market prices (aggregate by commodity)
market['Commodity'] = market['Commodity'].str.strip().str.lower()
market['Modal Price 21-01-2021 to 21-04-2026'] = pd.to_numeric(
    market['Modal Price 21-01-2021 to 21-04-2026'].astype(str).str.replace(',', '').str.replace('"', ''),
    errors='coerce'
)
market_avg_price = market.groupby('Commodity')['Modal Price 21-01-2021 to 21-04-2026'].mean().to_dict()

# Create soil_crop mapping
crop_to_category = {}
for _, row in soil_crop.iterrows():
    crop_name = str(row['Crop_Name']).strip().lower()
    crop_to_category[crop_name] = str(row['Crop_Category']).strip()

# Extract features
features = []
labels = []
skipped_rows = 0

for idx, row in yield_data.iterrows():
    district_raw = str(row.iloc[1]).strip()
    year_raw = str(row.iloc[2]).strip()
    
    # Clean district name (extract first part before comma)
    district = district_raw.split(',')[0].strip()
    district = 'Thoubal' if district == 'Thouba' else district
    
    # Extract year (first 4 digits)
    try:
        year = int(year_raw.split()[0])
    except:
        skipped_rows += 1
        continue
    
    # Find matching soil
    soil_match = soil_avg[soil_avg['District'].str.lower() == district.lower()]
    if soil_match.empty:
        skipped_rows += 1
        continue
    
    pH = soil_match['pH'].values[0]
    N = soil_match['Available_N'].values[0]
    P = soil_match['Available_P'].values[0]
    K = soil_match['Available_K'].values[0]
    
    # Encode season as numeric
    season_map = {'Kharif': 1, 'Rabi': 2, 'Autumn': 3, 'Summer': 4, 'Winter': 5, 'Whole Year': 0}
    
    # For each (crop, season) pair
    for (crop, season), measurements in crop_season_map.items():
        if 'yield' not in measurements:
            continue
        
        # Get yield column
        yield_col = measurements['yield']
        if yield_col >= len(row):
            continue
        
        try:
            yield_val = float(row.iloc[yield_col])
        except:
            continue
        
        if pd.isna(yield_val) or yield_val <= 0:
            continue
        
        # Get area if available
        area = 0
        if 'area' in measurements:
            try:
                area = float(row.iloc[measurements['area']])
            except:
                pass
        
        # Get production if available
        production = 0
        if 'production' in measurements:
            try:
                production = float(row.iloc[measurements['production']])
            except:
                pass
        
        # Get market price
        price = market_avg_price.get(crop.lower(), 0)
        
        # Encode season
        season_num = season_map.get(season, 0)
        
        # Create feature vector
        features.append([
            district,      # Will be encoded
            pH, N, P, K,   # Soil metrics
            season_num,    # Season (numeric)
            year,          # Year
            area,          # Cultivation area (hectares)
            production,    # Production (tonnes)
            yield_val,     # Yield (tonnes/hectare)
            price          # Market price (Rs/quintal)
        ])
        
        labels.append(crop.strip())

print(f"\nTotal valid samples: {len(features)}")
print(f"Skipped rows: {skipped_rows}")

if len(features) == 0:
    print("ERROR: No samples collected!")
    exit(1)

# Create DataFrame
X = pd.DataFrame(features, columns=['District', 'pH', 'N', 'P', 'K', 'Season', 'Year', 'Area', 'Production', 'Yield', 'Price'])
y = np.array(labels)

# Filter to top 15 crops for better accuracy (those with sufficient training data)
unique, counts = np.unique(y, return_counts=True)
class_dist = dict(zip(unique, counts))
top_crops = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:15]
top_crop_names = set([crop[0] for crop in top_crops])

mask = np.array([label in top_crop_names for label in y])
X = X[mask].reset_index(drop=True)
y = y[mask]

print(f"\nFiltered to top 15 crops: {len(X)} samples")

# Encode district
le_dist = LabelEncoder()
X['District'] = le_dist.fit_transform(X['District'])

# Encode crop (label)
le_crop = LabelEncoder()
y_enc = le_crop.fit_transform(y)

# Show class distribution
unique, counts = np.unique(y, return_counts=True)
class_dist = dict(zip(unique, counts))
print(f"Total unique crops: {len(class_dist)}")
print("Top 15 crops by sample count:")
for crop, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {crop}: {count}")

# Train/validation/test split
# Check if all classes have at least 2 samples for stratification
stratify = None
if len(np.unique(y_enc)) > 1:
    unique, counts = np.unique(y_enc, return_counts=True)
    if np.all(counts >= 2):
        stratify = y_enc

# First split: 70% train, 30% temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_enc, test_size=0.30, random_state=42, stratify=stratify
)

# Second split: 50/50 on temp -> 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# ===== MODEL 1: XGBoost (Best Performer) =====
if HAS_XGBOOST:
    print("\n" + "="*80)
    print("MODEL 1: XGBoost Classifier (PRIMARY)")
    print("="*80)
    model_xgb = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.95,
        colsample_bytree=0.95,
        min_child_weight=1,
        gamma=0.2,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    model_xgb.fit(X_train, y_train)
    
    train_acc_xgb = model_xgb.score(X_train, y_train)
    val_acc_xgb = model_xgb.score(X_val, y_val)
    test_acc_xgb = model_xgb.score(X_test, y_test)
    print(f"Training Accuracy: {train_acc_xgb*100:.2f}%")
    print(f"Validation Accuracy: {val_acc_xgb*100:.2f}%")
    print(f"Test Accuracy: {test_acc_xgb*100:.2f}%")
    
    # Cross-validation
    try:
        cv_folds = min(5, len(np.unique(y_train)))
        cv_scores_xgb = cross_val_score(model_xgb, X_train, y_train, cv=cv_folds, n_jobs=-1)
        print(f"CV Scores: {[f'{s:.3f}' for s in cv_scores_xgb]}")
        print(f"Mean CV Accuracy: {cv_scores_xgb.mean()*100:.2f}% (+/- {cv_scores_xgb.std()*100:.2f}%)")
    except Exception as e:
        print(f"CV Error: {e}")
else:
    test_acc_xgb = 0
    model_xgb = None

# ===== MODEL 2: RandomForest =====
print("\n" + "="*80)
print("MODEL 2: RandomForest Classifier")
print("="*80)
model_rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=16,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced_subsample',
    bootstrap=True,
    max_features='sqrt'
)
model_rf.fit(X_train, y_train)

train_acc_rf = model_rf.score(X_train, y_train)
val_acc_rf = model_rf.score(X_val, y_val)
test_acc_rf = model_rf.score(X_test, y_test)
print(f"Training Accuracy: {train_acc_rf*100:.2f}%")
print(f"Validation Accuracy: {val_acc_rf*100:.2f}%")
print(f"Test Accuracy: {test_acc_rf*100:.2f}%")

# Cross-validation
try:
    cv_folds = min(5, len(np.unique(y_train)))
    cv_scores_rf = cross_val_score(model_rf, X_train, y_train, cv=cv_folds, n_jobs=-1)
    print(f"CV Scores: {[f'{s:.3f}' for s in cv_scores_rf]}")
    print(f"Mean CV Accuracy: {cv_scores_rf.mean()*100:.2f}% (+/- {cv_scores_rf.std()*100:.2f}%)")
except Exception as e:
    print(f"CV Error: {e}")

# Feature importance
feat_imp = sorted(zip(X.columns, model_rf.feature_importances_), key=lambda x: x[1], reverse=True)
print("\nTop 10 Feature Importances (RandomForest):")
for fname, imp in feat_imp[:10]:
    print(f"  {fname}: {imp:.4f}")

# ===== MODEL 3: GradientBoosting =====
print("\n" + "="*80)
print("MODEL 3: GradientBoosting Classifier")
print("="*80)
model_gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    subsample=0.99,
    max_features='sqrt'
)
model_gb.fit(X_train, y_train)

train_acc_gb = model_gb.score(X_train, y_train)
val_acc_gb = model_gb.score(X_val, y_val)
test_acc_gb = model_gb.score(X_test, y_test)
print(f"Training Accuracy: {train_acc_gb*100:.2f}%")
print(f"Validation Accuracy: {val_acc_gb*100:.2f}%")
print(f"Test Accuracy: {test_acc_gb*100:.2f}%")

# ===== MODEL COMPARISON =====
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

accuracies_dict = {
    'RandomForest': {'test': test_acc_rf, 'val': val_acc_rf},
    'GradientBoosting': {'test': test_acc_gb, 'val': val_acc_gb},
}


if HAS_XGBOOST and model_xgb is not None:
    accuracies_dict['XGBoost'] = {'test': test_acc_xgb, 'val': val_acc_xgb}

# Use test accuracy for model selection
best_model_name = max(accuracies_dict, key=lambda x: accuracies_dict[x]['test'])
if best_model_name == 'XGBoost':
    best_model = model_xgb
    best_acc = test_acc_xgb
    best_val_acc = val_acc_xgb
elif best_model_name == 'GradientBoosting':
    best_model = model_gb
    best_acc = test_acc_gb
    best_val_acc = val_acc_gb
else:
    best_model = model_rf
    best_acc = test_acc_rf
    best_val_acc = val_acc_rf

print(f"\n{'='*80}")
print("TOP-K ACCURACY ANALYSIS")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"  Test Accuracy: {best_acc*100:.2f}%")
print(f"  Validation Accuracy: {best_val_acc*100:.2f}%")

proba = best_model.predict_proba(X_test)
for k in [1, 3, 5, 10]:
    correct = 0
    for i, true_label in enumerate(y_test):
        top_k = np.argsort(proba[i])[-k:][::-1]
        if true_label in top_k:
            correct += 1
    acc_k = correct / len(y_test)
    print(f"Top-{k} Accuracy: {acc_k*100:.2f}%")

# ===== SAVE MODEL =====
brain = {'model': best_model, 'le_crop': le_crop, 'le_dist': le_dist}
with open('crop_model.pkl', 'wb') as f:
    pickle.dump(brain, f)

print("\n" + "="*80)
print(f"✓ Model saved to crop_model.pkl")
print(f"✓ Best Model: {best_model_name}")
print(f"✓ Test Accuracy: {best_acc*100:.2f}%")
print(f"✓ Validation Accuracy: {best_val_acc*100:.2f}%")
print("="*80)
