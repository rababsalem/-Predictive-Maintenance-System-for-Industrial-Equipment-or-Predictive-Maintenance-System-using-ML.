import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 1. تحميل وتجهيز البيانات
columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
          [f'sensor{i}' for i in range(1, 22)]

file_path = r'E:\Predictive Maintenance\CMaps\train_FD001.txt'
train = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)

max_cycle = train.groupby('engine_id')['cycle'].max().reset_index()
max_cycle.columns = ['engine_id', 'max_cycle']
train = train.merge(max_cycle, on='engine_id', how='left')
train['RUL'] = (train['max_cycle'] - train['cycle']).clip(upper=125)
train.drop('max_cycle', axis=1, inplace=True)

drop_cols = ['setting3', 'sensor1', 'sensor5', 'sensor6', 'sensor10', 
             'sensor16', 'sensor18', 'sensor19']
features = [col for col in train.columns if col not in drop_cols + ['engine_id', 'cycle', 'RUL']]

# 2. تقسيم وتدريب النموذج
unique_engines = train['engine_id'].unique()
train_ids, val_ids = train_test_split(unique_engines, test_size=0.2, random_state=42)

val_data = train[train['engine_id'].isin(val_ids)].copy()

X_train = train[train['engine_id'].isin(train_ids)][features]
y_train = train[train['engine_id'].isin(train_ids)]['RUL']
X_val = val_data[features]
y_val = val_data['RUL']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# 3. حفظ التنبؤات في جدول Dataframe
val_data['predicted_RUL'] = model.predict(X_val_scaled)
val_data['error'] = (val_data['RUL'] - val_data['predicted_RUL']).abs()

# 4. إنشاء قاعدة بيانات SQLite وتخزين البيانات
engine_db = create_engine('sqlite:///E:/Predictive Maintenance/predictive_maintenance.db')

results_df = val_data[['engine_id', 'cycle', 'RUL', 'predicted_RUL', 'error']]
results_df.columns = ['engine_id', 'cycle', 'actual_RUL', 'predicted_RUL', 'error']

results_df.to_sql('predictions', engine_db, if_exists='replace', index=False)
print("تم حفظ البيانات في قاعدة البيانات predictive_maintenance.db بنجاح!\n")

# 5. تنفيذ استعلامات SQL حقيقية
print("--- [SQL 1] أكثر 5 محركات بها نسبة خطأ في التنبؤ ---")
query_high_error = """
SELECT engine_id, cycle, actual_RUL, predicted_RUL, error
FROM predictions
ORDER BY error DESC
LIMIT 5
"""
print(pd.read_sql(query_high_error, engine_db))

print("\n--- [SQL 2] متوسط الخطأ لكل محرك ---")
query_avg_error = """
SELECT engine_id, AVG(error) as avg_error, COUNT(cycle) as total_cycles
FROM predictions
GROUP BY engine_id
ORDER BY avg_error ASC
LIMIT 5
"""
print(pd.read_sql(query_avg_error, engine_db))