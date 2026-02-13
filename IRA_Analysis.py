#!/usr/bin/env python
# coding: utf-8

# In[11]:


import yfinance as yf
import pandas as pd
import numpy as np
import time

def get_clean_data(ticker, start_date):
    # Try up to 3 times if the server fails
    for i in range(3):
        try:
            df = yf.download(ticker, start=start_date, multi_level_index=False)
            if not df.empty:
                # Force it to a flat Series of floats
                return df['Close'].astype(float).copy()
        except Exception as e:
            print(f"Attempt {i+1} failed for {ticker}. Retrying...")
            time.sleep(2) # Wait 2 seconds before retrying
    return None

# 1. Download with the new stable function
voo = get_clean_data("VOO", "2010-01-01")
tnx = get_clean_data("^TNX", "2010-01-01")

if voo is not None and tnx is not None:
    # 2. IRA Macro Calculations
    ma200 = voo.rolling(window=200).mean()
    dist_200 = (voo - ma200) / ma200

    # 20-day percentage change for interest rates
    tnx_change = tnx.pct_change(periods=20) 

    # 3. Target: 1-Year Forward Return (252 Trading Days)
    y_long = (voo.shift(-252) > voo).astype(int)

    # 4. Final Feature Table
    X_long = pd.DataFrame({
        'Trend_Health': dist_200,
        'Rate_Pressure': tnx_change
    }).dropna()

    # Sync y with X
    y_long = y_long.loc[X_long.index]
    print("--- SUCCESS: IRA Macro Data Loaded ---")
    print(X_long.tail())
else:
    print("Download failed after retries. Check your internet connection!")


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Split (shuffle=False is CRITICAL for time-series/IRA data)
X_train, X_test, y_train, y_test = train_test_split(X_long, y_long, test_size=0.2, shuffle=False)

# 2. Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_test_scaled = scaler.transform(X_test).astype('float32')

# 3. Convert targets to float32 for the GPU
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

print(f"Training on data from {X_train.index[0].year} to {X_train.index[-1].year}")
print(f"Testing on data from {X_test.index[0].year} to {X_test.index[-1].year}")


# In[13]:


import tensorflow as tf

# Define a stable architecture
model_ira = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'), 
    tf.keras.layers.BatchNormalization(), # Keeps training stable
    tf.keras.layers.Dropout(0.1),         # Prevents "memorizing" specific years
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Gives us a % Probability
])

model_ira.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

# Train the model
# We use more epochs because macro trends move slowly
history = model_ira.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)


# In[14]:


# 1. Grab the absolute latest data point (Today)
latest_macro = X_long.tail(1)

# 2. Scale it using the training scaler
latest_scaled = scaler.transform(latest_macro).astype('float32')

# 3. Get the "Bullish Probability" for the next year
ira_probability = model_ira.predict(latest_scaled)[0][0]

# 4. Final Readout
# --- NEW & IMPROVED SIGNAL MESSAGES ---
print(f"--- 12-MONTH IRA OUTLOOK (Feb 2026) ---")
print(f"AI Confidence Score: {ira_probability:.2%}")

if ira_probability > 0.85:
    print("SIGNAL: ALL SYSTEMS GO. This is a rare, high-conviction time to add money to your IRA. The 'Macro Ocean' is very calm.")

elif ira_probability > 0.65:
    print("SIGNAL: STEADY GROWTH. Stick to your plan. If you have extra cash, it's a good time to put it to work.")

elif ira_probability > 0.45:
    print("SIGNAL: WAIT AND WATCH. The market is 'choppy.' Don't make big moves; just let your current investments sit.")

else:
    print("SIGNAL: DEFENSIVE MODE. The AI sees a storm. Instead of buying stocks, consider keeping your new contributions in 'Cash/Interest' for a few months.")


# In[15]:


# Save the 'Brain'
model_ira.save('ira_macro_model_v1.keras')

# Save the 'Scaler' (You need this to process future data exactly the same way)
import joblib
joblib.dump(scaler, 'ira_scaler_v1.pkl')

print("Model and Scaler saved! You are officially an AI Quant.")


# In[16]:


import matplotlib.pyplot as plt

# Create the 200-day Moving Average
voo_df = yf.download("VOO", period="2y")
voo_df['MA200'] = voo_df['Close'].rolling(200).mean()

plt.figure(figsize=(12, 6))
plt.plot(voo_df['Close'], label='VOO Price', color='blue', alpha=0.5)
plt.plot(voo_df['MA200'], label='200-Day Moving Average', color='red', lw=2)

# Highlight where the price is above/below the trend
plt.title("S&P 500 Trend Health (Model Input Analysis)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("This visualization demonstrates the 'Trend Health' feature used by the AI model.")


# In[ ]:




