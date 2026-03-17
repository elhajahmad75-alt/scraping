import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv('ebay_tech_deals.csv')

# =========================
# 2. CLEAN DATA
# =========================

# Clean price
df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Clean original price
df['original_price'] = df['original_price'].str.replace('$', '', regex=False).str.replace(',', '')
df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')

# Clean shipping (extract numbers, free = 0)
df['shipping'] = df['shipping'].str.extract(r'(\d+\.?\d*)')
df['shipping'] = pd.to_numeric(df['shipping'], errors='coerce').fillna(0)

# Drop missing important rows
df.dropna(subset=['price', 'original_price'], inplace=True)

# =========================
# 3. FEATURE ENGINEERING
# =========================

# Price features
df['effective_price'] = df['price'] + df['shipping']
df['discount_ratio'] = (df['original_price'] - df['price']) / df['original_price']
df['discount_percentage'] = df['discount_ratio'] * 100

# Title features
df['title_len'] = df['title'].astype(str).apply(len)
df['title_word_count'] = df['title'].astype(str).apply(lambda x: len(x.split()))

df['has_new'] = df['title'].str.contains('new', case=False, na=False).astype(int)
df['has_used'] = df['title'].str.contains('used', case=False, na=False).astype(int)
df['has_refurbished'] = df['title'].str.contains('refurbished', case=False, na=False).astype(int)
df['has_bundle'] = df['title'].str.contains('bundle', case=False, na=False).astype(int)
df['has_case'] = df['title'].str.contains('case', case=False, na=False).astype(int)
df['has_charger'] = df['title'].str.contains('charger', case=False, na=False).astype(int)

# Time features
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['hour'] = df['timestamp'].dt.hour.fillna(0)
df['weekday'] = df['timestamp'].dt.weekday.fillna(0)

# =========================
# 4. TARGET VARIABLE
# =========================
df['high_discount'] = df['discount_percentage'] > 20

# =========================
# 5. SAVE FEATURES FILE
# =========================
df.to_csv('ebay_features.csv', index=False)

# =========================
# 6. PREPARE DATA
# =========================

baseline_features = ['price', 'original_price']

extended_features = [
    'price', 'original_price', 'effective_price', 'discount_ratio',
    'title_len', 'title_word_count',
    'has_new', 'has_used', 'has_refurbished',
    'has_bundle', 'has_case', 'has_charger',
    'hour', 'weekday'
]

X_base = df[baseline_features]
X_ext = df[extended_features]
y = df['high_discount']

X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y)

X_train_e, X_test_e, _, _ = train_test_split(
    X_ext, y, test_size=0.2, random_state=42, stratify=y)

# =========================
# 7. TRAIN MODELS
# =========================

model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train_b, y_train)
y_pred1 = model1.predict(X_test_b)

model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train_e, y_train)
y_pred2 = model2.predict(X_test_e)

model3 = DecisionTreeClassifier(random_state=42)
model3.fit(X_train_e, y_train)
y_pred3 = model3.predict(X_test_e)

# =========================
# 8. EVALUATION FUNCTION
# =========================

def evaluate(y_test, y_pred, name):
    return f"""
{name}
Accuracy: {accuracy_score(y_test, y_pred)}
Precision: {precision_score(y_test, y_pred)}
Recall: {recall_score(y_test, y_pred)}
F1 Score: {f1_score(y_test, y_pred)}
Confusion Matrix:
{confusion_matrix(y_test, y_pred)}
"""

# =========================
# 9. SAVE RESULTS
# =========================

results = ""
results += evaluate(y_test, y_pred1, "Logistic Regression (Baseline)")
results += evaluate(y_test, y_pred2, "Logistic Regression (Extended)")
results += evaluate(y_test, y_pred3, "Decision Tree")

# =========================
# 10. HANDLE IMBALANCE
# =========================

majority = df[df['high_discount'] == False]
minority = df[df['high_discount'] == True]

balanced = pd.concat([
    majority.sample(len(minority), random_state=42),
    minority
])

X_bal = balanced[extended_features]
y_bal = balanced['high_discount']

X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

model_bal = DecisionTreeClassifier(random_state=42)
model_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = model_bal.predict(X_test_bal)

results += evaluate(y_test_bal, y_pred_bal, "Balanced Decision Tree")

# Save to file
with open("model_results.txt", "w") as f:
    f.write(results)

print("✅ Done! Files generated:")
print("- ebay_features.csv")
print("- model_results.txt")

