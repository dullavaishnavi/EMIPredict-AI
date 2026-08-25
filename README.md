# EMIPredict AI — Streamlit Deployment

This repository contains the Streamlit deployment for the EMIPredict AI project.

## 1. Project structure

```text
EMIPredict_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── best_classification_model.joblib
│   └── best_regression_model.joblib
└── data/
    └── emi_prediction_dataset.csv   # optional
```

## 2. Get the trained models

Run the Google Colab notebook first:

`EMIPredict_AI_End_to_End.ipynb`

At the end, it creates:

- `best_classification_model.joblib`
- `best_regression_model.joblib`

Download the `EMIPredict_models.zip`, extract it, and place the two `.joblib`
files inside the `models/` folder.

## 3. Test locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `models/best_classification_model.joblib`
   - `models/best_regression_model.joblib`
3. Open Streamlit Community Cloud.
4. Connect your GitHub account.
5. Select the repository.
6. Select `app.py` as the main file.
7. Deploy.

## 5. Application pages

- Dashboard
- EMI Prediction
- About the Model

The prediction page uses the exact feature-engineering logic from the Colab
project before passing the customer data into the saved pipelines.

## Important

Do not claim that the project's target accuracy (>90%) or RMSE (<₹2,000)
has been achieved until the Colab notebook has actually been executed and
the measured results are recorded.
