import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Mengaktifkan MLflow Autolog
mlflow.sklearn.autolog()
print('MLflow Autolog telah diaktifkan')

# 2. Memuat dataset
df = pd.read_csv('used_cars_preprocessed.csv')
print('Data berhasil dimuat')

# Pisahkan fitur (X) dan target (y)
X = df.drop(columns=['price'])
y = df['price']

# 3. Data preprocessing (lanjutan)
# ubah fitur kategorikal menjadi numerik dengan One-Hot Encoding
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Membuat preprocessor menggunakan ColumnTransformer
# OneHotEncoder akan diterapkan ke kolom kategorikal
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Skip kolom numerik
)

# 4. Pembagian data (train & test split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Ukuran data training: {X_train.shape}")
print(f"Ukuran data testing: {X_test.shape}")

# 5. Membuat dan melatih model menggunakan Pipeline
# Gunakan RandomForestRegressor karena cocok untuk regresi

# definisikan model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# membuat pipeline: preprocessing -> model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Mulai MLflow logging
with mlflow.start_run() as run:
    # melatih model
    print("Memulai pelatihan model...")
    pipeline.fit(X_train, y_train)
    print("Pelatihan selesai.")

    # lakukan prediksi
    y_pred = pipeline.predict(X_test)

    # evaluasi metrik
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print('\nMetrik Evaluasi')
    print(f'RMSE: {rmse:.2f}')
    print(f'R2 Score: {r2:.4f}')

    # simpan model secara eksplisit untuk memastikan model
    # tersimpan di bawah artifak run
    mlflow.sklearn.log_model(pipeline, "random_forest_model")

    # ambil Run ID untuk keperluan dokumentasi
    run_id = run.info.run_id
    print(f"\nModel dan metrik disimpan secara lokal oleh MLflow di folder 'mlruns'.")
    print(f'Run ID: {run_id}\n')