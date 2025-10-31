# Automated COVID-19 Detection from Chest X-rays

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/code/muhamedsalah1000/svm-notebook)

## Overview

A machine learning pipeline for automated COVID-19 detection from chest X-ray images, achieving **85% accuracy** using an optimized stacking ensemble approach.

### Key Highlights
- **Dataset:** COVIDqu (COVID-19, Non-COVID, Normal classes)
- **Approach:** Stacking ensemble with HOG features
- **Models:** XGBoost + Random Forest + SVM with Logistic Regression meta-learner
- **Performance:** 85% accuracy with balanced metrics across all classes

## Quick Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| COVID-19 | 0.87 | 0.85 | 0.86 |
| Non-COVID | 0.84 | 0.87 | 0.85 |
| Normal | 0.83 | 0.83 | 0.83 |
| **Overall** | **0.85** | **0.85** | **0.85** |

## Installation

```bash
pip install numpy pandas scikit-learn xgboost opencv-python scikit-image
```

## Usage

```python
from pipeline import COVIDDetectionPipeline

# Initialize and train
pipeline = COVIDDetectionPipeline(image_size=(128, 128))
pipeline.fit(X_train, y_train)

# Predict
prediction = pipeline.predict(image)
```

## Project Structure

```
├── data/               # COVIDqu dataset
├── notebooks/          # Kaggle notebook
├── models/             # Trained models
├── pipeline.py         # Main pipeline code
└── MLpaper.pdf         # Full research paper
```

## Documentation

📄 **For complete methodology, results, and analysis, please refer to [MLpaper.pdf](MLpaper.pdf)**

The paper includes:
- Detailed preprocessing and segmentation steps
- Complete model evolution (62% → 85% accuracy)
- Confusion matrix analysis
- Related work and comparative studies
- Future research directions

## Contributors

**Cairo University - Systems and Biomedical Engineering Department**

- **Hager Ehab** - Data preprocessing, lung segmentation, paper writing
- **Shimaa Kamel** - PCA implementation, baseline classifiers
- **Ahmed Rafaat** - Memory optimization, advanced ensembles
- **Muhammed Salah** - Hyperparameter tuning, final model optimization

## Resources

- 📊 [Kaggle Notebook](https://www.kaggle.com/code/muhamedsalah1000/svm-notebook)
- 📄 [Research Paper](MLpaper.pdf)

## Citation

```bibtex
@article{rafaat2025covid19detection,
  title={Automated COVID-19 Detection from Chest X-rays Using Machine Learning Pipelines},
  author={Rafaat, Ahmed and Salah, Muhammed and Kamel, Shimaa and Ehab, Hager},
  journal={Systems and Biomedical Engineering, Cairo University},
  year={2025}
}
```

---

**Note:** This is a screening tool designed to assist healthcare professionals, not replace professional medical diagnosis.
