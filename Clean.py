import pandas as pd
import numpy as np
import os
from scipy.stats.mstats import winsorize

# Set working directory to current folder
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the dataset
file_path = "FIES PUF 2012 Vol.1.csv"
df = pd.read_csv(file_path, encoding="latin1", low_memory=False)

# Step 1: Standardize Column Names
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

# Step 2: Handle Missing Values
# Drop columns with excessive missing values (e.g., more than 50%)
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

# Define ALL categorical columns (both text-based and numeric)
categorical_columns = [
    # Text-based categorical
    "W_REGN", "URB", "SEX", "MS",  'W_OID', "JOB", "HHTYPE", "TENURE", "TOILET", "ELECTRIC", "WATER",
    # Numeric categorical 
    "OCCUP", "KB", "CW", "AGELESS5", "AGE5_17", "EMPLOYED_PAY", "EMPLOYED_PROF"
]

# ✅ Convert categorical columns to category type BEFORE any transformation
df[categorical_columns] = df[categorical_columns].astype("category")

# 📌 Backup categorical data before processing
categorical_original_data = df[categorical_columns].copy()

# Step 3: Fill missing values in numeric columns (EXCLUDING categorical columns)
numeric_columns = ['W_SHSN', 'NATDC', 'HSE_ALTERTN', 'PSU', 'REGPC', 'T_ACTRENT',
 'T_BIMPUTED_RENT', 'T_RENTVAL', 'BLDG_TYPE', 'RFACT', 'FSIZE', 'WALLS',
 'BWEIGHT', 'AGRI_SAL', 'RSTR', 'NONAGRI_SAL', 'T_IMPUTED_RENT', 'AGE',
 'T_FOOD_NEC', 'MEMBERS', 'NATPC', 'ROOF', 'FOOD_ACCOM_SRVC', 'W_HCN',
 'POP_ADJ', 'SPOUSE_EMP', 'REGDC', 'HGC']

# Convert only these columns to numeric
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Fill missing numeric values with median
df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.median()), axis=0)

# Step 4: Fill asset ownership columns with 0 (but exclude any that are categorical)
asset_columns = [
    "DISTANCE", "RADIO_QTY", "TV_QTY", "CD_QTY", "STEREO_QTY", "REF_QTY", "WASH_QTY", 
    "AIRCON_QTY", "CAR_QTY", "LANDLINE_QTY", "CELLPHONE_QTY", "PC_QTY", "OVEN_QTY", 
    "MOTOR_BANCA_QTY", "MOTORCYCLE_QTY"
]
df[asset_columns] = df[asset_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Step 5: Remove Duplicates
df.drop_duplicates(inplace=True)

# Step 6: Convert Income and Expenditure columns to float (excluding categorical columns)
income_expenditure_columns = [
    "WAGES", "NETSHARE", "CASH_ABROAD", "CASH_DOMESTIC", "RENTALS_REC", "INTEREST", "PENSION", "DIVIDENDS",
    "OTHER_SOURCE", "NET_RECEIPT", "REGFT", "NET_CFG", "NET_LPR", "NET_FISH", "NET_FOR", "NET_RET", "NET_MFG",
    "NET_COM", "NET_TRANS", "NET_MIN", "NET_CONS", "NET_NEC", "EAINC", "TOINC", "LOSSES", "T_BREAD", "T_MEAT",
    "T_FISH", "T_MILK", "T_OIL", "T_FRUIT", "T_VEG", "T_SUGAR", "T_COFFEE", "T_MINERAL", "T_ALCOHOL", "T_TOBACCO",
    "T_OTHER_VEG", "T_FOOD_HOME", "T_FOOD_OUTSIDE", "T_FOOD", "T_CLOTH", "T_FURNISHING", "T_HEALTH", "T_HOUSING_WATER",
    "T_TRANSPORT", "T_COMMUNICATION", "T_RECREATION", "T_EDUCATION", "T_MISCELLANEOUS", "T_OTHER_EXPENDITURE",
    "T_OTHER_DISBURSEMENT", "T_NFOOD", "T_TOTEX", "T_TOTDIS", "T_OTHREC", "T_TOREC", "PCINC"
]
df[income_expenditure_columns] = df[income_expenditure_columns].apply(pd.to_numeric, errors='coerce')



# Step 9: Final cleanup
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

#Categorical Stuffz
# First, explicitly add '0' and '7' as categories to OCCUP, KB, CW
df['OCCUP'] = df['OCCUP'].cat.add_categories([0])
df['KB'] = df['KB'].cat.add_categories([0])
df['CW'] = df['CW'].cat.add_categories([0, 7])

# Change 0s in CW to 7 before filling NaNs
df['CW'] = pd.to_numeric(df['CW'], errors='coerce')
df.loc[df['CW'] == 0, 'CW'] = 7

# Fill NaNs in OCCUP and KB with 0 (meaning no occupation/business)
df[['OCCUP', 'KB']] = df[['OCCUP', 'KB']].fillna(0)

# For CW, fill NaNs with 0 (Not Applicable)
df['CW'] = df['CW'].fillna(0)

# Now explicitly set CW=0 where OCCUP and KB are both 0 (indicating no occupation/business)
df.loc[(df['OCCUP'] == 0) & (df['KB'] == 0), 'CW'] = 0

# Remove unused categories (cleanup)
df['OCCUP'] = df['OCCUP'].cat.remove_unused_categories()
df['KB'] = df['KB'].cat.remove_unused_categories()


# Optional: check the unique values after transformation
print("Updated unique values after fixes:")
print(f"OCCUP: {df['OCCUP'].unique()}")
print(f"KB: {df['KB'].unique()}")
print(f"CW: {df['CW'].unique()}")

# Fill NaNs in AGELESS5, AGE5_17, EMPLOYED_PAY, EMPLOYED_PROF with 0 explicitly
age_employment_columns = ['AGELESS5', 'AGE5_17', 'EMPLOYED_PAY', 'EMPLOYED_PROF']

# First, ensure that categories include '0' explicitly
for col in age_employment_columns:
    if 0 not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories([0])

# Replace NaN with 0
df[age_employment_columns] = df[age_employment_columns].fillna(0)

# Optional: remove unused categories for neatness
for col in age_employment_columns:
    df[col] = df[col].cat.remove_unused_categories()

# Quick verification
for col in age_employment_columns:
    print(f"{col} unique values: {df[col].unique()}")


# Step 8: Apply Log Transformation to Numeric Columns (EXCLUDING categorical columns)
log_transformed_columns = {}

for col in income_expenditure_columns:
    min_val = df[col].min()
    if min_val < 0:
        # Shift data to positive if negative values are present
        df[col] = np.log1p(df[col] - min_val)
    else:
        df[col] = np.log1p(df[col])
    
# Step 10: Save the Cleaned Dataset
cleaned_file_path = "FIES_2012_Cleaned.csv"
df.to_csv(cleaned_file_path, index=False)


