# ==========================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# 1. Class Imbalance Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df_clean, x='Is_International', palette='Set2')
plt.title('Target Class Distribution (0: Domestic, 1: International)')
plt.xlabel('Visitor Category')
plt.ylabel('Count')
plt.savefig('target_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Visit Hours Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data=df_clean, x='Hour', hue='Is_International', multiple='stack', bins=24, palette='Set1')
plt.title('Distribution of Visit Hours: Domestic vs International')
plt.xlabel('Visit Hour (0-23)')
plt.ylabel('Frequency')
plt.savefig('visit_hours_distribution.png', dpi=300, bbox_inches='tight')
plt.show()