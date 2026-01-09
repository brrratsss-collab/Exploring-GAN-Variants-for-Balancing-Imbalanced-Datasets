# Exploring GAN Variants for Balancing Imbalanced Datasets (Tabular)

This repo implements **Vanilla GAN + two GAN variants (WGAN-GP, LSGAN)** to generate **minority-class synthetic samples** for an imbalanced fraud-detection dataset (`card_transdata.csv`). It then balances the training set and evaluates a downstream classifier across scenarios.

## Files
- `GAN_Imbalance_Project_Notebook.ipynb` – main notebook (recommended)
- `run_experiment.py` – script version (optional)
- `GAN_Imbalance_Report.pdf` – 4–6 page report (generated example)

## Quick start (Notebook)
1. Install requirements (example):
   ```bash
   pip install numpy pandas scikit-learn matplotlib torch reportlab
   ```
2. Put `card_transdata.csv` in the same folder as the notebook.
3. Open and run `GAN_Imbalance_Project_Notebook.ipynb`.

## Quick start (Script)
```bash
python run_experiment.py --data card_transdata.csv --sample_size 20000 --gan_epochs 200
```

### Key arguments
- `--sample_size`: set to `0` to use the full dataset (may be slow)
- `--gan_epochs`: increase for better synthetic quality (recommended 200–500+)
- `--variant`: which GAN variant to use for the main comparison: `wgan_gp` or `lsgan`

## Notes
- GANs are trained on the **minority class only**.
- Synthetic samples are generated **only for the training split** to avoid test leakage.
- For tabular data, binary features are rounded back to {0,1} after inverse scaling.

## Demo video (3–5 min)
Suggested outline:
1) Show dataset imbalance plot  
2) Explain GANs implemented (Vanilla, WGAN-GP, LSGAN)  
3) Show generated sample distribution plots  
4) Show evaluation table + metric comparison plot  
