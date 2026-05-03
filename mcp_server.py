from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Initialize the MCP Server
mcp = FastMCP("HousePricePredictor")

# ---------------------------------------------------------
# Tool 1: Dataset Profiling (Critical for checking skewness/nulls)
# ---------------------------------------------------------
@mcp.tool()
def analyze_dataset_profile(filepath: str, target_column: str = "SalePrice") -> str:
    """
    Loads a CSV dataset and returns critical metadata: column types, missing values, 
    and the skewness of the target variable to determine if log-transformation is needed.
    """
    try:
        df = pd.read_csv(filepath)
        profile = {
            "shape": df.shape,
            "missing_values_summary": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "numerical_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        if target_column in df.columns:
            profile["target_skewness"] = float(df[target_column].skew())
            
        return json.dumps(profile, indent=2)
    except Exception as e:
        return f"Error reading dataset: {str(e)}"

# ---------------------------------------------------------
# Tool 2: Preprocessing Blueprint Generator
# ---------------------------------------------------------
@mcp.tool()
def suggest_column_transformer_blueprint(filepath: str) -> str:
    """
    Analyzes the dataset and returns a recommended scikit-learn ColumnTransformer 
    strategy, separating features into ordinal, nominal, and continuous pipelines.
    """
    df = pd.read_csv(filepath)
    # Simple logic to categorize for the AI
    continuous = df.select_dtypes(include=[np.number]).nunique()[lambda x: x > 15].index.tolist()
    ordinal = ["OverallQual", "OverallCond"] # Hardcoded domain knowledge for Ames
    nominal = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    blueprint = {
        "continuous_features_to_scale": [c for c in continuous if c not in ['Id', 'SalePrice']],
        "ordinal_features_to_encode": ordinal,
        "nominal_features_to_onehot": [n for n in nominal if n not in ordinal]
    }
    return json.dumps(blueprint, indent=2)

# ---------------------------------------------------------
# Tool 3: Model Evaluator
# ---------------------------------------------------------
@mcp.tool()
def evaluate_predictions(y_true_json: str, y_pred_json: str, is_log_transformed: bool = False) -> str:
    """
    Calculates MSE, RMSE, MAE, and R2. 
    If is_log_transformed is True, it applies np.expm1 before calculating metrics
    to ensure the error represents actual dollar amounts.
    """
    y_true = np.array(json.loads(y_true_json))
    y_pred = np.array(json.loads(y_pred_json))
    
    if is_log_transformed:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
        
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }
    return json.dumps(metrics, indent=2)

if __name__ == "__main__":
    # Runs the server over stdio, ready to be attached to an MCP Client
    mcp.run(transport='stdio')