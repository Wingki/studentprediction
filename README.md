# Predicting University Student Performance (Demo)

This app demonstrates a workflow for predicting student course performance (pass/fail) using academic, behavioral, and socio-demographic data.  
It uses machine learning with interpretable models and a Streamlit dashboard.

## Usage

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Run model training and interpretation:
    ```
    python src/train.py
    ```
3. Launch the dashboard:
    ```
    streamlit run src/app.py
    ```

- Demo data provided (`data/students.csv`). Replace with real data for production.
- All outputs and plots are saved into `data/`.