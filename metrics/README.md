# Model metrics contract

`model_metrics.json` is the file Streamlit reads to render the Metrics module.

The file can be regenerated from the TimeSformer and VideoMAE runs with:

```powershell
python scripts/import_transformer_runs.py
```

Current imported source:

```text
C:\Users\SEBAS\PycharmProjects\YoloModel\runs_timesformer_cctv_clean
C:\Users\SEBAS\PycharmProjects\YoloModel\runs_videomae_cctv_clean
```

For each trained model, export:

- `summary`: `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`, `pr_auc`.
- `confusion_matrix`: four rows with `actual`, `predicted`, and `count`.
- `history`: one row per epoch with `train_loss`, `val_loss`, `train_accuracy`, and `val_accuracy`.
- `roc_curve`: rows with `fpr` and `tpr`.
- `pr_curve`: rows with `recall` and `precision`.
- `model_path`: path to the trained model weights.

The current values for TimeSformer and VideoMAE are imported from the local training runs. ROC/PR AUC values are real, but ROC/PR curve points are estimated in the Streamlit charts unless prediction scores are exported.
