import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_scale(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df

if __name__ == "__main__":
    X_scaled, df = load_and_scale("/content/sample_data/Mall_Customers.csv")
    print(f"Scaled data shape: {X_scaled.shape}")