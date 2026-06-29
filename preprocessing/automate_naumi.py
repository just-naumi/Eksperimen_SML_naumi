import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    # Define paths
    input_path = os.path.join(os.path.dirname(__file__), '..', 'namadataset_raw', 'credit_risk_dataset.csv')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'namadataset_preprocessing')
    output_path = os.path.join(output_dir, 'data_bersih.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}. Please make sure the dataset is placed correctly.")
        return
        
    print("Data loaded successfully. Starting preprocessing...")
    
    # 1. Handling Missing Values
    print("Handling missing values...")
    # Impute missing values for numerical columns with median
    if 'person_emp_length' in df.columns:
        df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    if 'loan_int_rate' in df.columns:
        df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
        
    # 2. Removing Outliers
    print("Removing outliers...")
    if 'person_age' in df.columns:
        df = df[df['person_age'] <= 100]
    if 'person_emp_length' in df.columns:
        df = df[df['person_emp_length'] <= 60]
        
    # 3. Encoding Categorical Variables
    print("Encoding categorical variables...")
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # 4. Scaling Numerical Features (excluding the target 'loan_status')
    print("Scaling numerical features...")
    target_col = 'loan_status'
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
        
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # 5. Save the cleaned dataset
    print(f"Saving cleaned dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing()
