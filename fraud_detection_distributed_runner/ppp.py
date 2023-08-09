import pandas as pd

from service import svc
from numpy import array

for runner in svc.runners:
    runner.init_local(quiet=True)

test_transactions: pd.DataFrame = pd.read_csv("./data/test_transaction.csv")[:500]
sample_json = test_transactions.to_dict(orient="records")

response = svc.apis["is_fraud"].func(test_transactions)

print(len(response["is_fraud"]))