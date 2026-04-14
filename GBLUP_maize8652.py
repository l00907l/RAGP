import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

SEED = 1

print("Loading data...")

# Genotype features
with open("./example-data/maize8652/X.pkl", "rb") as f:
    x_data = pickle.load(f)

# Phenotypes
y1 = pd.read_csv("./example-data/maize8652/SNP_pca/DTT.csv")  # DTT
y2 = pd.read_csv("./example-data/maize8652/SNP_pca/PH.csv")   # PH
y3 = pd.read_csv("./example-data/maize8652/SNP_pca/EW.csv")   # EW

# Rename phenotype columns
y1.columns = ["DTT"]
y2.columns = ["PH"]
y3.columns = ["EW"]

# Convert genotype matrix to DataFrame if needed
if not isinstance(x_data, pd.DataFrame):
    x_data = pd.DataFrame(x_data)
    x_data.columns = [f"PC_{i}" for i in range(x_data.shape[1])]

# Combine genotype and phenotype data
data = pd.concat([x_data, y1, y2, y3], axis=1)


def run_gblup_prediction(data, trait_name, feature_cols, seed):
    train, test = train_test_split(data, test_size=0.1, random_state=seed)

    X_train = train[feature_cols].values
    y_train = train[trait_name].values

    X_test = test[feature_cols].values
    y_test = test[trait_name].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], cv=5)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    pcc, _ = pearsonr(y_test, y_pred)

    return pcc, model.alpha_


# Maize8652 phenotype columns
phenotype_cols = ["DTT", "PH", "EW"]
feature_cols = [c for c in data.columns if c not in phenotype_cols]

print(f"{'Trait':<10} | {'Accuracy (PCC)':<15} | {'Best Alpha':<15}")
print("-" * 45)

results = {}

for trait in phenotype_cols:
    temp_data = data.dropna(subset=[trait])

    pcc, best_alpha = run_gblup_prediction(temp_data, trait, feature_cols, SEED)
    results[trait] = pcc

    print(f"{trait:<10} | {pcc:.4f}          | {best_alpha:.1f}")

print("-" * 45)
print("Done. These results correspond to the GBLUP baseline on Maize8652.")
