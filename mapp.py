import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Define paths for raw and processed data
RAW_DATA_PATH = "malaria_indicators_btn.csv"
PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE_PATH = PROCESSED_DIR / "processed_malaria_data.csv"

# --- Step 1: Data Loading and Initial Validation ---

def load_data(file_path):
    """Loads the CSV file and performs initial validation."""
    print(f"Loading data from: {file_path}")
    try:
        # Load the CSV file. Assuming the second row contains meaningful headers,
        # we will use `header=1` to skip the first header row if necessary,
        # but often it's best to inspect the file structure first.
        # Based on the snippet, the second row is the intended header.
        df = pd.read_csv(file_path, header=1)
        print("Initial data shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return None

def validate_data_structure(df):
    """Checks for missing values and correct data types."""
    print("\n--- Initial Validation Check ---")

    # 1. Simplify and clean column names
    df.columns = df.columns.str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip().str.replace(r'\s+', '_', regex=True)
    df = df.rename(columns={'gho_display': 'indicator_name', 'year_display': 'year', 'country_display': 'country', 'numeric': 'value_num'})

    # Drop columns with near-identical or irrelevant information for this scope (e.g., URLs, regional codes)
    cols_to_drop = [col for col in df.columns if any(keyword in col for keyword in ['code', 'url', 'startyear', 'endyear', 'region', 'dimension', 'low', 'high', 'value'])]
    # Keep only `value_num` and relevant descriptive columns
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df = df.rename(columns={'value_num': 'malaria_cases_or_rate'})

    # Filter for necessary columns and remove Bhutan-specific columns (as the whole file is Bhutan)
    df = df[['indicator_name', 'year', 'malaria_cases_or_rate']].copy()

    # Convert 'year' to integer type
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    print("Columns after cleaning:", df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values (Initial Check):")
    print(df.isnull().sum())

    return df

# --- Step 2: Preprocessing ---

def preprocess_data(df):
    """Handles missing values, and prepares data for EDA/ML."""
    print("\n--- Preprocessing Data ---")

    # 1. Missing Value Handling (malaria_cases_or_rate is the key metric)
    # Strategy: Impute missing numerical values with the median.
    # We will assume a simple imputation based on the overall dataset, but for production,
    # a more sophisticated approach (e.g., grouping by indicator_name) would be better.
    imputer = SimpleImputer(strategy='median')
    df['malaria_cases_or_rate'] = imputer.fit_transform(df[['malaria_cases_or_rate']])

    # 2. Outlier Detection (Simple IQR for demonstration)
    Q1 = df['malaria_cases_or_rate'].quantile(0.25)
    Q3 = df['malaria_cases_or_rate'].quantile(0.75)
    IQR = Q3 - Q1
    # Define a simple outlier boundary
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Cap outliers (Winsorization) to prevent extreme values from skewing models
    df['malaria_cases_or_rate'] = np.where(df['malaria_cases_or_rate'] > upper_bound, upper_bound, df['malaria_cases_or_rate'])
    df['malaria_cases_or_rate'] = np.where(df['malaria_cases_or_rate'] < lower_bound, lower_bound, df['malaria_cases_or_rate'])

    # 3. Encoding of Categorical Variables (Indicator_Name is the primary category)
    # We'll pivot the data to create a wide format suitable for analysis.
    # Each unique indicator becomes a new feature/column.
    df_pivot = df.pivot_table(index='year', columns='indicator_name', values='malaria_cases_or_rate', aggfunc='first').reset_index()

    print(f"Pivoted data shape: {df_pivot.shape}")
    print("New columns (Indicator Names):", df_pivot.columns.tolist())

    # After pivot, there might be new missing values (years without data for a specific indicator).
    # Fill these new NaNs (missing years for an indicator) using ffill or bfill.
    df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill')

    # 4. Feature Scaling (on the newly created indicator columns)
    # Scaling is crucial if we move to models like k-NN or Neural Networks.
    features_to_scale = df_pivot.columns.drop('year').tolist()
    scaler = StandardScaler()
    df_pivot[features_to_scale] = scaler.fit_transform(df_pivot[features_to_scale])

    print("\nMissing Values (Final Check):")
    print(df_pivot.isnull().sum().sum()) # Should be 0 after ffill/bfill

    return df_pivot

# --- Main Execution ---

if __name__ == "__main__":
    # Create the processed data directory if it doesn't exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and Validate
    raw_df = load_data(RAW_DATA_PATH)

    if raw_df is not None:
        cleaned_df = validate_data_structure(raw_df)

        # Step 2: Preprocess
        processed_df = preprocess_data(cleaned_df)

        # Store cleaned outputs
        processed_df.to_csv(PROCESSED_FILE_PATH, index=False)
        print(f"\nSuccessfully stored processed data to: {PROCESSED_FILE_PATH}")
        print("Head of processed data:")
        print(processed_df.head())
    else:
        print("Data processing halted due to loading error.")
