# iris_classify/profiling_bento.py
from datetime import datetime

import numpy as np
import numpy as np
import pandas as pd
from bentoml.client import Client

from service import IrisFeatures, Iris

random_input_size_fixed = IrisFeatures(
    features=[
        Iris(
            sepal_len=6.2,
            sepal_width=3.2,
            petal_len=5.2,
            petal_width=2.2,
        )
        for _ in range(0, 500) # 10 -> 1000
    ]
)


#test_transactions: pd.DataFrame = pd.read_csv("./data/test_transaction.csv")[:500]


# test code
client = Client.from_url("http://localhost:13000")

latency_list = []

for _ in range(500):
    t = datetime.now()
    res = client.call("classify", random_input_size_fixed)
    tt = datetime.now() - t
    latency_list.append(tt.total_seconds())

print(f"AVG: {sum(latency_list)/ len(latency_list)}")
print(f"Median: {np.median(sorted(latency_list,reverse=True))}")
print("percentile: ", np.percentile(latency_list, [50, 75, 100], interpolation='nearest'))


# dis-runner batch:1000
# AVG: 0.018282477999999984
# Median: 0.0155535
# percentile:  [0.012006 0.013273 0.01556  0.019609 0.119336]


# origin batch: 1000
# AVG: 0.017171670000000003
# Median: 0.015754499999999998
# percentile:  [0.013117 0.013989 0.015773 0.019711 0.044692]


# origin batch: 500
# AVG: 0.013489195999999998
# Median: 0.0116975
# percentile:  [0.010448 0.011255 0.011701 0.016609 0.06671 ]

# dis-runner batch:500
# AVG: 0.013557794000000001
# Median: 0.011887
# percentile:  [0.010025 0.01126  0.011895 0.016824 0.040127]

# origin batch: 1000
# AVG: 0.01884337000000001
# Median: 0.018504
# percentile:  [0.012582 0.013627 0.01852  0.019785 0.112543]

# dis-runner batch:1000
# AVG: 0.015433535999999998
# Median: 0.013708999999999999
# percentile:  [0.012226 0.013114 0.013711 0.01771  0.046506]

# origin batch: 2000
# AVG: 0.023584341999999994
# Median: 0.0242135
# percentile:  [0.017471 0.023354 0.024216 0.02479  0.047753]

# dis-runner batch:2000
# AVG: 0.020371260000000016
# Median: 0.0215655
# percentile:  [0.015666 0.016767 0.021569 0.022553 0.04615 ]






# dis-runner batch:500
# AVG: 0.016225917999999985
# Median: 0.013294
# percentile:  [0.011216 0.012087 0.013309 0.017662 0.112076]

# origin batch:500
# AVG: 0.016365810000000015
# Median: 0.014929000000000001
# percentile:  [0.010949 0.011779 0.014956 0.017485 0.101747]

# origin batch:1000
# AVG: 0.018189206000000003
# Median: 0.019429000000000002
# percentile:  [0.013301 0.01466  0.01943  0.019853 0.042245]

# dis-runner batch:1000
# AVG: 0.01900408199999999
# Median: 0.0190215
# percentile:  [0.012927 0.01399  0.019023 0.019575 0.113427]