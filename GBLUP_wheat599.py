import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

SEED = 1

print("Loading data...")

with open("./example-data/wheat599/SNP_pca/wheat599_pc95.pkl", "rb") as f:
    x_data = pickle.load(f)

y1 = pd.read_csv("./example-data/wheat599/SNP_pca/wheat1.Y")
y2 = pd.read_csv("./example-data/wheat599/SNP_pca/wheat2.Y")
y3 = pd.read_csv("./example-data/wheat599/SNP_pca/wheat3.Y")
y4 = pd.read_csv("./example-data/wheat599/SNP_pca/wheat4.Y")

# Rename phenotype columns
y1.columns = ["env1"]
y2.columns = ["env2"]
y3.columns = ["env3"]
y4.columns = ["env4"]

# Convert genotype matrix to DataFrame if needed
if not isinstance(x_data, pd.DataFrame):
    x_data = pd.DataFrame(x_data)
    x_data.columns = [f"PC_{i}" for i in range(x_data.shape[1])]

# Combine genotype and phenotype data
data = pd.concat([x_data, y1, y2, y3, y4], axis=1)


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


# Wheat599 phenotype columns
phenotype_cols = ["env1", "env2", "env3", "env4"]
feature_cols = [c for c in data.columns if c not in phenotype_cols]

print(f"{'Env':<10} | {'Accuracy (PCC)':<15} | {'Best Alpha':<15}")
print("-" * 45)

results = {}

for trait in phenotype_cols:
    temp_data = data.dropna(subset=[trait])

    pcc, best_alpha = run_gblup_prediction(temp_data, trait, feature_cols, SEED)
    results[trait] = pcc

    print(f"{trait:<10} | {pcc:.4f}          | {best_alpha:.1f}")

print("-" * 45)
print("Done. These results correspond to the GBLUP baseline on Wheat599.")
