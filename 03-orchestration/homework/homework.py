import pandas as pd
import pickle
from datetime import datetime, timedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow
from prefect import flow, task
from prefect import get_run_logger
from prefect.task_runners import SequentialTaskRunner


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@task
def get_paths(date=None):
    logger = get_run_logger()
    val_date = date - timedelta(days=date.day)
    logger.info(f"Date of validation data: {val_date}")
    val_path = f"./data/fhv_tripdata_{val_date.strftime('%Y-%m')}.parquet"
    train_path = f"./data/fhv_tripdata_{(val_date - timedelta(days=val_date.day)).strftime('%Y-%m')}.parquet"
    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    if not date:
        date = datetime.today()
    else:
        date = datetime.strptime(date, '%Y-%m-%d')
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    with open(f"models/model-{date.strftime('%Y-%m-%d')}.bin", "wb") as f_out:
        pickle.dump(lr, f_out)
    mlflow.log_artifact(f"models/model-{date.strftime('%Y-%m-%d')}.bin", artifact_path="models_mlflow")

    with open(f"models/dv-{date.strftime('%Y-%m-%d')}.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact(f"models/dv-{date.strftime('%Y-%m-%d')}.b", artifact_path="preprocessor")

    run_model(df_val_processed, categorical, dv, lr)

#main(date="2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(
        cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
