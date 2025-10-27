# Context-Aware Network Intelligence System (DATATHON 2025 Submission)
# Project Overview: Proactive Congestion and Anomaly Management
This project addresses the challenge of managing highly volatile time-series network traffic (VLAN 106 and VLAN 06) by building a Context-Aware Dual-Output Deep Learning Model. We move beyond reactive alerting to provide preemptive Congestion Forecasting and forensic Anomaly Classification.
# Core Deliverables
Predictive Model: LSTM Sequence-to-Sequence for congestion forecasting (Mb/s).
Inference Model: STL Decomposition combined with Isolation Forest for anomaly identification and classification (DDoS vs. Massive Download).
Contextual Integration: Quantification of the relationship between network activity and Academic Schedule seasonality.
# Methodology & Technical Stack
1. Data Source & Preparation
Data: 5 months of 5-minute interval traffic data (March - May 2025) for two VLANs (Wired + Wireless metrics).
Features: Engineered Total_Mb volume, Input/Output Balance, and Packet-to-Bit Ratio (to infer traffic type/packet size).
Key Insight: EDA confirmed Wednesday as the peak-demand day for VLAN106 and Friday for VLAN06 and also observed a significant traffic decline correlating with the end of the semester.
2. Congestion Forecasting (Predictive Model)
Model: Sequence-to-Sequence LSTM. Trained on 4 hours (48 timesteps) of multivariate data to predict the next 5-minute traffic volume.
Evaluation (Regression): Confirmed high performance:
Mean Absolute Error (MAE): 10.02 Mb/s (Average forecast error).
Evaluation (Classification): Alerting when traffic exceeds the 90th percentile (106.31 Mb/s):
Accuracy: 0.95
F1-Score: 0.79 (Balanced accuracy of alerts).
Precision: 0.76
Recall: 83.65% (Effectiveness in catching actual congestion events).
3. Anomaly Classification (Inference Model)
Detection: Used STL Decomposition to isolate the unmodeled traffic (the Residual) from the predictable Trend and Seasonality.
Classification: Applied Isolation Forest coupled with Rule-Based Inference to classify the most severe anomalies:
Massive Download: Low Packet-to-Bit Ratio (large packets).
DDoS/Scan: High Packet-to-Bit Ratio (small packets) + high input imbalance.
Result: Successfully classified 206 high-risk outliers, distinguishing benign high-volume transfers from complex network anomalies.
# Running the Project
# Prerequisites
Python 3.8+
The raw network data file tstat_data1.xlsx .
# Installation
pip install tensorflow scikit-learn pandas numpy matplotlib openpyxl joblib statsmodels

# File Structure
VLAN_Traffic_Analysis.ipynb (Jupyter Notebook): Contains the complete end-to-end code for Data Loading, EDA, Feature Engineering, STL Decomposition, Model Training, and Evaluation.
lstm_congestion_model.h5: The saved LSTM model weights and architecture.
scaler.pkl: The saved MinMaxScaler object (critical for preprocessing new data).
# Model Deployment (Example Usage)
To reload the model and predict traffic for a new 4-hour window, use the deployment script:
python model_deployment.py




