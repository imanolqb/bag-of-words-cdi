# Bag of Words - Text Classification Project

Classification of scientific articles using classical Machine Learning approaches (Bag of Words with TF-IDF) and Deep Learning with PyTorch.

## 📋 Project Description

This project implements text classification to predict which journal a scientific article belongs to based on its content. Two main approaches are compared:

- **Classical ML**: TF-IDF vectorization + sklearn classifiers (Logistic Regression, Linear SVM, Naive Bayes, SVC)
- **Deep Learning**: PyTorch neural networks with mean pooling

## 🗂️ Project Structure

```
TrabajoDeCurso/
├── data/
│   ├── raw/              # Original JSON files by journal
│   └── processed/        # Processed CSV dataset
├── notebooks/
│   ├── 01_build_dataset.ipynb          # Data preprocessing
│   ├── 02_train_sklearn.ipynb          # Classical ML models
│   ├── 03_train_pytorch.ipynb          # Deep Learning models
│   └── 04_models_comparison.ipynb      # Model comparison
├── src/                  # Python scripts (notebook equivalents)
├── models/               # Trained models (.joblib, .pt)
├── docs/                 # Results and LaTeX report
└── requirements.txt      # Python dependencies
```

## 🎯 Objectives

1. Compare classical ML vs. Deep Learning approaches for text classification
2. Analyze the impact of class imbalance handling techniques:
   - `class_weight='balanced'`
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Hybrid approach (SMOTE + class_weight)
3. Evaluate models using Macro F1-score (appropriate for imbalanced datasets)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- CUDA (optional, for GPU acceleration with PyTorch)

### Installation

```bash
# Clone the repository
git clone https://github.com/imanolqb/bag-of-words-cdi.git
cd bag-of-words-cdi

# Install dependencies
pip install -r requirements.txt
```

### Usage

Run the notebooks in order:

1. **01_build_dataset.ipynb**: Load and process JSON files into unified CSV
2. **02_train_sklearn.ipynb**: Train classical ML models
3. **03_train_pytorch.ipynb**: Train Deep Learning models
4. **04_models_comparison.ipynb**: Compare all approaches
   
## 🛠️ Technologies

- **Data Processing**: pandas, numpy
- **Classical ML**: scikit-learn, imbalanced-learn
- **Deep Learning**: PyTorch
- **Visualization**: matplotlib, seaborn
- **Documentation**: Jupyter Notebook, LaTeX

## 📝 Class Imbalance Handling

The dataset shows significant class imbalance:

- Journal 1: 866 articles
- Journal 2: 1,865 articles
- Journal 3: 7,671 articles
- Journal 5: 3,572 articles

Techniques implemented:
- ✅ `class_weight='balanced'`
- ✅ SMOTE oversampling
- ✅ Hybrid (SMOTE + class_weight)
- ✅ Macro F1-score evaluation

## 📄 License

This project is part of an academic assignment for the Master's program at ULPGC.

## 👤 Author

**Imanol**
- GitHub: [@imanolqb](https://github.com/imanolqb)

## 🙏 Acknowledgments

- Master SIANI - Universidad de Las Palmas de Gran Canaria
- Course: Clasificación y Detección de Imágenes (CDI)
