import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. تحميل البيانات المعالجة
columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
          [f'sensor{i}' for i in range(1, 22)]

file_path = r'E:\Predictive Maintenance\CMaps\train_FD001.txt'
train = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)

# حساب الـ RUL مع Clipping
max_cycle = train.groupby('engine_id')['cycle'].max().reset_index()
max_cycle.columns = ['engine_id', 'max_cycle']
train = train.merge(max_cycle, on='engine_id', how='left')
train['RUL'] = (train['max_cycle'] - train['cycle']).clip(upper=125)
train.drop('max_cycle', axis=1, inplace=True)

# 2. حذف الاعمدة الثابتة تماماً في FD001 (لا تحمل أي تباين مفيد للتعلم)
drop_cols = ['setting3', 'sensor1', 'sensor5', 'sensor6', 'sensor10', 
             'sensor16', 'sensor18', 'sensor19']

features = [col for col in train.columns if col not in drop_cols + ['engine_id', 'cycle', 'RUL']]

# 3. تقسيم البيانات بناءً على المحركات (Engine-level Split) لمنع تسريب البيانات
unique_engines = train['engine_id'].unique()
train_ids, val_ids = train_test_split(unique_engines, test_size=0.2, random_state=42)

train_data = train[train['engine_id'].isin(train_ids)]
val_data = train[train['engine_id'].isin(val_ids)]

X_train = train_data[features]
y_train = train_data['RUL']

X_val = val_data[features]
y_val = val_data['RUL']

# 4. توحيد المقاييس (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 5. تدريب نموذج Random Forest
print("جاري تدريب النموذج...")
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# 6. التنبؤ وتقييم الأداء
y_pred = model.predict(X_val_scaled)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print("\n--- نتائج تقييم النموذج ---")
print(f"Mean Absolute Error (MAE) : {mae:.2f} cycles")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f} cycles")
print(f"R² Score : {r2:.3f}")