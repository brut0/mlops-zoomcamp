from datetime import datetime
import pandas as pd
import batch

from deepdiff import DeepDiff


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    df = pd.DataFrame(data, columns=columns)
    categorical = ['PUlocationID', 'DOlocationID']

    expected_output = pd.DataFrame(
        columns=columns + ["duration"],
        data=[
            ("-1", "-1", dt(1, 2), dt(1, 10), 8.0),
            ("1", "1", dt(1, 2), dt(1, 10), 8.0),
        ],
    )
    print(expected_output.head())

    actual_output = batch.prepare_data(df, categorical)
    print(actual_output.head())

    diff = DeepDiff(actual_output.to_dict(), expected_output.to_dict(), ignore_order=True, significant_digits=3)
    print(diff)
    assert 'type_changes' not in diff.keys()
    assert 'values_changed' not in diff.keys()
    