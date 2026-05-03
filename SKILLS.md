# Project Skills & Competencies

This project demonstrates a comprehensive, end-to-end applied machine learning workflow, focusing heavily on rigorous data preprocessing, algorithm selection, and model evaluation for tabular data.

### 📊 Exploratory Data Analysis (EDA) & Statistical Profiling
* **Automated Data Profiling:** Utilizing `AutoViz` for rapid, high-dimensional data visualization to identify underlying distributions and anomalies.
* **Target Variable Transformation:** Identifying right-skewness in continuous target variables and applying logarithmic transformations to normalize distributions and optimize for Root Mean Squared Logarithmic Error (RMSLE).
* **Multicollinearity Analysis:** Leveraging `seaborn` and `matplotlib` to design correlation heatmaps, identifying and mitigating highly correlated independent variables.

### 🛠️ Advanced Data Engineering & Preprocessing
* **Pipeline Orchestration:** Architecting robust, leak-proof data transformations using `scikit-learn`'s `Pipeline` and `ColumnTransformer`.
* **Contextual Imputation:** Handling high-sparsity datasets (79 features) by applying domain-aware imputation strategies (e.g., differentiating between a structural missing value and a true null).
* **Feature Scaling & Encoding:** Selectively applying `StandardScaler` and `MinMaxScaler` for numerical continuous data, and strategically deploying `OneHotEncoder` and `OrdinalEncoder` for nominal and ordinal categorical data.

### 🤖 Machine Learning & Predictive Modeling
* **Algorithm Implementation:** Training and evaluating tree-based models (`RandomForestRegressor`, `XGBRegressor`) and distance-based algorithms (`KNeighborsRegressor`).
* **Ensemble Learning:** Designing a composite predictive model using `VotingRegressor` to aggregate the strengths of diverse base estimators and reduce overall variance.
* **Hyperparameter Optimization:** Executing systematic grid searches (`GridSearchCV`) to fine-tune model parameters across multidimensional spaces.

### 📐 Methodological Evaluation & Validation
* **Cross-Validation:** Implementing `KFold` cross-validation to ensure model generalizability and prevent overfitting on training subsets.
* **Performance Metrics:** Evaluating models using standard regression metrics including Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared ($R^2$), tailored to the specific business context of the predictions.

### 💻 Software Engineering for Data Science
* **Modular Notebook Design:** Structuring Jupyter/Colab environments with clean, logical, and sequential cell execution.
* **Documentation & Reproducibility:** Maintaining strict markdown documentation within the analytical environment to explain the *why* behind algorithmic choices, ensuring the research is highly reproducible.