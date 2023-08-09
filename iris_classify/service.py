# iris_classify/service.py
import asyncio
from typing import TypedDict

import numpy as np
import pandas as pd

import bentoml
from bentoml.io import JSON, PandasDataFrame
from bentoml.io import NumpyNdarray

# from line_profiler_pycharm import profile




class Iris(TypedDict):
    sepal_len: float
    sepal_width: float
    petal_len: float
    petal_width: float


class IrisFeatures(TypedDict):
    features: list[Iris]


# origin
iris_clf_runner1 = bentoml.sklearn.get("iris_clf_with_feature_names:latest").to_runner(name="iris_clf_runner1")

svc = bentoml.Service("iris_classifier_pydantic", runners=[iris_clf_runner1])

@svc.api(input=JSON(), output=NumpyNdarray())
async def classify(iris_features: TypedDict) -> np.ndarray:
    iris_features_list = iris_features["features"]

    input_data = np.array([list(aa.values()) for aa in iris_features_list])
    result1 = await iris_clf_runner1.predict.async_run(input_data)
    return result1



# distributed-runner
iris_clf_runner1 = bentoml.sklearn.get("iris_clf_with_feature_names:latest").to_runner(name="iris_clf_runner1")
iris_clf_runner2 = bentoml.sklearn.get("iris_clf_with_feature_names:latest").to_runner(name="iris_clf_runner2")

svc = bentoml.Service("iris_classifier_pydantic", runners=[iris_clf_runner1,iris_clf_runner2])

@svc.api(input=JSON(), output=NumpyNdarray())
async def classify(iris_features: TypedDict) -> np.ndarray:
    iris_features_list = iris_features["features"]

    # Convert list to an array
    input_data = np.array([list(aa.values()) for aa in iris_features_list])
    result1, result2 = await asyncio.gather(
        iris_clf_runner1.predict.async_run(input_data[:250]),
        iris_clf_runner2.predict.async_run(input_data[250:]),
    )
    return np.concatenate((result1, result2), axis=0)

