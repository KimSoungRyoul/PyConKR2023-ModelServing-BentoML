import asyncio

import numpy as np
import pandas as pd
from sample import sample_input

import bentoml
from bentoml.io import JSON
from bentoml.io import PandasDataFrame
#from line_profiler_pycharm import profile

model_ref = bentoml.xgboost.get("ieee-fraud-detection-lg:latest")
preprocessor = model_ref.custom_objects["preprocessor"]
fraud_model_runner1 = model_ref.to_runner(name="fraud_model_runner1")
fraud_model_runner2 = model_ref.to_runner(name="fraud_model_runner2")

svc = bentoml.Service("fraud_detection", runners=[fraud_model_runner1, fraud_model_runner2])

input_spec = PandasDataFrame.from_sample(sample_input)


@svc.api(input=input_spec, output=JSON())
#@profile
async def is_fraud(input_df: pd.DataFrame):
    input_df = input_df.astype(sample_input.dtypes)


    # results1 =  fraud_model_runner1.predict_proba.run(input_features[:250])
    # results2 =  fraud_model_runner2.predict_proba.run(input_features[250:])
    results1, results2 = await asyncio.gather(
        fraud_model_runner1.predict_proba.async_run( preprocessor.transform(input_df[:250])),
        fraud_model_runner2.predict_proba.async_run( preprocessor.transform(input_df[250:])),
    )
    results = np.concatenate((results1, results2), axis=0)
    predictions = np.argmax(results, axis=1)  # 0 is not fraud, 1 is fraud
    return {"is_fraud": list(map(bool, predictions)), "is_fraud_prob": results[:, 1]}
