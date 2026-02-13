import tensorflow as tf
import joblib
import yfinance as yf
import pandas as pd

# 1. LOAD SAVED ASSETS
model = tf.keras.models.load_model('ira_macro_model_v1.keras')
scaler = joblib.load('ira_scaler_v1.pkl')

# 2. FETCH CURRENT DATA
voo = yf.download("VOO", period="1y")['Close']
tnx = yf.download("^TNX", period="1y")['Close']

# 3. CALCULATE CURRENT FEATURES
current_trend = (voo.iloc[-1] - voo.rolling(200).mean().iloc[-1]) / voo.rolling(200).mean().iloc[-1]
current_rates = tnx.pct_change(20).iloc[-1]

# 4. PREDICT
input_data = pd.DataFrame([[current_trend, current_rates]])
input_scaled = scaler.transform(input_data)
prob = model.predict(input_scaled, verbose=0)[0][0]

# 5. HUMAN-READABLE OUTPUT
print(f"\n--- 12-Month IRA Strategy Signal ---")
print(f"AI Confidence Score: {prob:.2%}")

if prob > 0.85:
    print("SIGNAL: STRONG ACCUMULATE. High-conviction entry point.")
elif prob > 0.60:
    print("SIGNAL: MAINTAIN. Stay invested, standard contributions.")
else:
    print("SIGNAL: DEFENSIVE. Consider holding new cash in SPAXX (Money Market).")
