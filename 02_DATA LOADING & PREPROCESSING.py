# ==========================================
# STEP 2: DATA LOADING & PREPROCESSING
# ==========================================
# Read the dataset uploaded in Step 0
df = pd.read_csv(io.BytesIO(uploaded[file_name]))

# 1. Feature Engineering: Temporal Features
df['Time'] = pd.to_datetime(df['Time'])
df['Hour'] = df['Time'].dt.hour
df['DayOfWeek'] = df['Time'].dt.dayofweek

# 2. Define Target Class (1: International/Minority, 0: Domestic/Majority)
# Assuming there is a 'Country' column where 'Indonesia' is the domestic traffic
df['Is_International'] = np.where(df['Country'] == 'Indonesia', 0, 1)

# 3. Handle Missing Values in Referer
df['Referer'] = df['Referer'].fillna('Direct/Other')

# 4. Drop Explicit Geolocation to Prevent Data Leakage
columns_to_drop = ['Time', 'Visitor ID', 'City', 'Country', 'Keyword', 'Rank']
df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

print(f"Dataset shape after cleaning: {df_clean.shape}")