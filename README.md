# IRA Macro Regime Detector

### Overview
A Deep Learning model designed to identify high-probability market regimes for long-term IRA contributions. It predicts the probability of a positive 12-month return based on macro-economic and trend data.

### Core Logic
* **Target:** 1-year forward price appreciation.
* **Inputs:** 10-Year Treasury Yield ($^TNX$) and S&P 500 (VOO) Price/Trend distance.
* **Current Signal:** 91.4% Confidence (Strong Accumulate) as of Feb 2026.

### Project Files
* `train_model.py`: Script to download historical data and train the Neural Network.
* `check_signal.py`: Inference script to generate the current monthly score.
* `IRA_Analysis.ipynb`: Data visualization and exploratory analysis.
* `ira_macro_model_v1.keras`: Saved TensorFlow model weights.
* `ira_scaler_v1.pkl`: Saved Scikit-Learn scaler for data normalization.
* `requirements.txt`: List of necessary Python libraries.

### How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Generate current signal: `python check_signal.py`

### Tech Stack
* **AI:** TensorFlow / Keras (Neural Networks)
* **Data:** yFinance, Pandas, Numpy
* **Preprocessing:** Scikit-Learn (StandardScaler)
