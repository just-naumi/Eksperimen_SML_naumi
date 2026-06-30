import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

def run_preprocessing():
    print("Memulai proses preprocessing otomatis...")

    # 1. Load Data Mentah
    file_path = '../namadataset_raw/credit_risk_dataset.csv'
    df = pd.read_csv(file_path)
    
    # 2. Menangani Missing Values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # 3. Menghapus Data Duplikat
    df.drop_duplicates(inplace=True)

    # 4. Encoding Data Kategorikal
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # 5. Memisahkan Fitur dan Target & Normalisasi
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Buat DataFrame baru untuk data bersih
    df_cleaned = pd.DataFrame(X_scaled, columns=X.columns)
    df_cleaned['loan_status'] = y.values

    # 6. Menyimpan Data Bersih
    output_dir = 'namadataset_preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'data_bersih.csv')
    df_cleaned.to_csv(output_path, index=False)

    print(f"Data bersih berhasil diperbarui dan disimpan di: {output_path}")

if __name__ == "__main__":
    run_preprocessing()
#Done