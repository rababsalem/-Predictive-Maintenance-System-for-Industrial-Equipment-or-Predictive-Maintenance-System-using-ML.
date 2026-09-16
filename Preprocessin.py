import pandas as pd
import numpy as np

# 1. أسماء الأعمدة
columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
          [f'sensor{i}' for i in range(1, 22)]

# 2. تحديد المسار المطلق للملف
file_path = r'E:\Predictive Maintenance\CMaps\train_FD001.txt'

# 3. تحميل البيانات باستخدام المتغير file_path
train = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)

# 4. حساب أقصى دورة لكل محرك (وقت العطل)
max_cycle = train.groupby('engine_id')['cycle'].max().reset_index()
max_cycle.columns = ['engine_id', 'max_cycle']

# 5. دمج وحساب RUL
train = train.merge(max_cycle, on='engine_id', how='left')
train['RUL'] = train['max_cycle'] - train['cycle']

# 6. تطبيق Piecewise RUL Clipping (تحديد الحد الأقصى بـ 125 لدقة أفضل)
train['RUL'] = train['RUL'].clip(upper=125)

train.drop('max_cycle', axis=1, inplace=True)

# طباعة النتائج وحجم البيانات
print("--- أول 5 صفوف من البيانات ---")
print(train.head())
print("\nحجم البيانات (Rows, Columns):", train.shape)
print("\n--- معاينة RUL المحسوبة ---")
print(train[['engine_id', 'cycle', 'RUL']].head(10))