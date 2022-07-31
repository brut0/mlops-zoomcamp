import os
import sys
import pandas as pd
from datetime import datetime
import batch

from deepdiff import DeepDiff

sys.path.insert(1, os.path.abspath('.'))


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
df_input = pd.DataFrame(data, columns=columns)

endpoint_url = os.getenv('S3_ENDPOINT_URL', 'http://localhost:4566')
options = {
    'client_kwargs': {
        'endpoint_url': endpoint_url
    }
}
input_file = 's3://nyc-duration/in/2021-01.parquet'

df_input.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

os.system('python batch.py 2021 1')

actual_df = batch.read_data('test_{year:04d}-{month:02d}_result.parquet'.format(year=2021, month=1))
actual_dict = actual_df.to_dict(orient='records')

expected_dict = [{'ride_id': '2021/01_0', 'predicted_duration': 23.052085},
                 {'ride_id': '2021/01_1', 'predicted_duration': 46.236612}]

expected_df = pd.DataFrame(expected_dict)

diff = DeepDiff(actual_dict, expected_dict, significant_digits=3)


print('Actual', actual_df)
print('\n')
print('Expected', expected_df)
print('\n')
print('Difference', diff)

assert 'values_changed' not in diff
assert 'type_changes' not in diff

print('Success!')