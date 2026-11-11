# CodeAlpha_Credit_Scoring_Model

## Objective
Predict an individual's creditworthiness and assess default risk using past financial patterns.

## Methodology & Architecture
- **Features Analyzed:** Income, Total Debt, Debt-to-Income (DTI) ratio, Revolving Credit Utilization, Past Delinquencies.
- **Model Selected:** Random Forest Classifier with balanced class weighting.
- **Performance Evaluation:** Evaluated using Precision, Recall, F1-Score, and ROC-AUC.
- **Deployment:** Built with a native zero-dependency HTTP server delivering an interactive browser interface without external AVX wheel constraints.

## How to Run
1. Train and save model:
   ```bash
   python3 train.py
   ```
2. Start the web application:
   ```bash
   python3 server.py
   ```
3. Open `http://localhost:8080` in your browser.
