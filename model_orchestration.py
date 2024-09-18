## Import Libraries
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from zenml import Model, pipeline, step
from zenml.client import Client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from zenml.logger import get_logger

logger = get_logger(__name__)

# from sklearn.base import RegressorMixin
df_pth = 'https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv'
local_pth = "./datasets/SalaryData.csv"
local_model_register = "./model_performance.json"
local_model_pth = "./models"


@step(enable_cache=True)
def load_data() -> pd.DataFrame:
    if os.path.exists(local_pth):
        df = pd.read_csv(local_pth)
    else:
        os.makedirs("./datasets", exist_ok=True)
        df = pd.read_csv(df_pth)
        df.to_csv("./datasets/SalaryData.csv", index=False)
    return df


@step
def train_model(data: pd.DataFrame) -> Tuple[Model, Dict[str, float]]:
    y = data['Salary']
    X = data[['Experience Years']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=234)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Create a ZenML Model object
    zenml_model = Model(name="salary_prediction_model", model=model, metadata={"rmse": str(rmse)})

    return zenml_model, {"rmse": rmse}


@step
def promote_model(
        model: Model,
        metrics: Dict[str, float],
        stage: str = "production"
) -> bool:
    rmse = metrics["rmse"]

    # Get the ZenML client
    client = Client()

    try:
        # Try to get the previous production model
        previous_production_model = client.get_model_version(name=model.name, version="production")
        previous_production_rmse = float(previous_production_model.run_metadata["rmse"].value)
    except:
        # If there's no production model, set a high RMSE
        previous_production_rmse = float('inf')

    if rmse < previous_production_rmse:
        # Promote the model
        model.set_stage(stage, force=True)
        logger.info(f"Model promoted to {stage}!")
        return True
    else:
        logger.info(
            f"Model not promoted. Current RMSE ({rmse}) is not better than production RMSE ({previous_production_rmse})")
        return False


@pipeline
def simple_ml_pipeline():
    dataset = load_data()
    model, metrics = train_model(dataset)
    is_promoted = promote_model(model, metrics)

    return is_promoted


if __name__ == "__main__":
    run = simple_ml_pipeline()
