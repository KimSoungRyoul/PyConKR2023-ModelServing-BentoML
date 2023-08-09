import numpy as np
import pandas as pd
from line_profiler_pycharm import profile


@profile
def create_instance(sample_data):
    dataframe = pd.DataFrame(sample_data)
    dataframe[:5_000], dataframe[5_000:]

    arr = np.array(sample_data)
    arr[:5_000], arr[5_000:]


sample_data = [[1.111 for _ in range(0, 500)] for row_size in range(0, 1_000)]
for _ in range(0, 100):
    create_instance(sample_data)




