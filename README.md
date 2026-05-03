# Regression Project (Portfolio Edition)

This project demonstrates a complete, production-grade machine learning pipeline for house price prediction using the Ames Housing Dataset. It is designed as a portfolio piece to showcase advanced data science, engineering, and modern AI-powered development practices with GitHub Copilot.

---

## 🚀 Project Highlights

- **End-to-End ML Workflow:** From raw data to deployable model, with rigorous EDA, preprocessing, modeling, and evaluation.
- **Best Practices:** All transformations are pipeline-encapsulated, ensuring reproducibility and leak-proof validation.
- **Copilot-Driven Development:** Leveraged GitHub Copilot CLI, Notebooks, Chat, and PR features for code, documentation, and workflow automation.
- **Skills Demonstrated:** See [`SKILLS.md`](SKILLS.md) for a full breakdown of competencies and techniques applied.

---

## 🗂️ Project Structure

- `data/` — Raw and processed datasets
- `notebooks/` — Jupyter notebooks for EDA, modeling, and analysis
- `mcp_server.py` — Main Python script (if applicable)
- `train.csv`, `test.csv` — Ames Housing data splits
- `data_description.txt` — Detailed feature descriptions
- `SKILLS.md` — Portfolio skills and competencies summary

---

## 📊 Data & Features

- **Source:** Ames Housing Dataset (see `data_description.txt` for all 79 features)
- **Sample Features:**
  - `MSSubClass`: Dwelling type (e.g., 20=1-story 1946+, 60=2-story 1946+)
  - `MSZoning`: Zoning classification (RL, RM, etc.)
  - `LotFrontage`, `LotArea`: Lot dimensions
  - `Street`, `Alley`: Access types
  - ...and more

---

## 🧑‍💻 Workflow Overview

### 1. Exploratory Data Analysis (EDA)
- Automated profiling with AutoViz for rapid insights
- Custom seaborn/matplotlib plots for target distribution, feature relationships, and multicollinearity

### 2. Data Preprocessing
- Train/test split before any transformation
- Contextual imputation for missing values
- Feature scaling (Standard/MinMax)
- Categorical encoding (OneHot/Ordinal)
- All steps encapsulated in `Pipeline` and `ColumnTransformer`

### 3. Modeling & Tuning
- Log-transform of `SalePrice` for RMSLE optimization
- Model selection: RandomForest, KNN, XGBoost
- Hyperparameter tuning with `GridSearchCV` and `KFold` cross-validation
- Ensemble with `VotingRegressor`

### 4. Evaluation
- Metrics: RMSLE, RMSE, MAE, R²
- Visual diagnostics: prediction vs. actual plots

---

## 🛠️ Example Usage

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
X = df.drop('SalePrice', axis=1)
y = np.log1p(df['SalePrice'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = ['LotArea', 'LotFrontage']
cat_features = ['MSSubClass', 'MSZoning', 'Street']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestRegressor())
])
pipe.fit(X_train, y_train)
```

---

## 🤖 GitHub Copilot Integration

This project was built using the full suite of GitHub Copilot features:

- **Copilot CLI:** Automated code generation, documentation, and workflow scripting directly from the terminal.
- **Copilot Notebooks:** AI-powered code suggestions and markdown explanations in Jupyter/Colab, ensuring clear, reproducible analysis.
- **Copilot Chat:** Contextual code help, debugging, and best-practice advice within the IDE and terminal.
- **Copilot PRs:** AI-assisted pull request reviews, code suggestions, and auto-completion for collaborative development.
- **SKILLS.md:** Auto-generated skills summary, documenting all advanced techniques and tools applied in the project.

> **How Copilot Enhanced My Workflow:**
> - Accelerated boilerplate code and pipeline construction
> - Improved documentation and markdown clarity
> - Automated repetitive tasks (e.g., feature engineering templates)
> - Provided instant explanations and alternatives for ML techniques
> - Streamlined PR review and code quality

---

## 📝 Getting Started

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run notebooks in the `notebooks/` directory to reproduce results.

---

## 📦 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, autoviz

---

## 🤝 Contributing

Pull requests are welcome! Use GitHub Copilot features to streamline your workflow. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT License](LICENSE)

---

## 📬 Contact

For questions or suggestions, open an issue or contact the repository owner via GitHub.
