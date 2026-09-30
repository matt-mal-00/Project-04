# ==========================================
# STEP 4: ENCODING, DATA SPLITTING & SMOTE
# ==========================================
# 1. One-Hot Encoding for Categorical Features (Sparse Matrix Generation)
X = pd.get_dummies(df_clean.drop('Is_International', axis=1), drop_first=True)
y = df_clean['Is_International']

# 2. Stratified Data Splitting (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Apply SMOTE exclusively to the Training Set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"Original Train Class Distribution: \n{y_train.value_counts()}")
print(f"SMOTE Train Class Distribution: \n{pd.Series(y_train_smote).value_counts()}")