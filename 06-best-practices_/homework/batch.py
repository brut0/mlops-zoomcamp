#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd


S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")

with open("model.bin", "rb") as f_in:
    dv, lr = pickle.load(f_in)


def get_input_path(year, month):
    if S3_ENDPOINT_URL is None:
        input_path = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet"
    else:
        default_input_pattern = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
        input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
        input_path = input_pattern.format(year=year, month=month)
    return input_path


def get_output_path(year, month):
    if S3_ENDPOINT_URL is None:
        output_path = f"predict_{year:04d}-{month:02d}.parquet"
    else:
        default_output_pattern = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
        output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
        output_path = output_pattern.format(year=year, month=month)
    return output_path


def read_data(input_path: str) -> pd.DataFrame:
    if S3_ENDPOINT_URL is None:
        options = {}
    else:
        options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}

    df = pd.read_parquet(input_path, storage_options=options)

    return df


def prepare_data(df: pd.DataFrame, categorical) -> pd.DataFrame:
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def save_data(df: pd.DataFrame, filename: str):
    if S3_ENDPOINT_URL is None:
        options = {}
    else:
        options = {'client_kwargs': {'endpoint_url': S3_ENDPOINT_URL}}

    df.to_parquet(
        filename,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PUlocationID', 'DOlocationID']

    df = read_data(input_file)
    df = prepare_data(df, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file)



if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)