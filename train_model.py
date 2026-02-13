import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# 1. DATA INGESTION
# Downloading 15 years of S&P 500 and Interest Rate data
voo = yf.download("VOO", start="2010-01-01")['Close']
tnx = yf.download("^TNX", start="2010-01-01")['Close']

# 2. FEATURE ENGINEERING
data = pd.DataFrame({
    'Trend_Health': (voo - voo.rolling(200).mean()) / voo.rolling(200).mean(),
    'Rate_Pressure': tnx.pct_change(20),
    'Target': (voo.shift(-252) > voo).astype(int) # 1-year forward return
}).dropna()

# 3. PREPROCESSING
scaler = StandardScaler()
X = scaler.fit_transform(data[['Trend_Health', 'Rate_Pressure']])
y = data['Target']

# 4. MODEL ARCHITECTURE
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 5. TRAINING & SAVING
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=32, verbose=0)

# SAVE THE BRAIN AND THE GLASSES
model.save('ira_macro_model_v1.keras')
joblib.dump(scaler, 'ira_scaler_v1.pkl')
print("Model and Scaler successfully saved!")
