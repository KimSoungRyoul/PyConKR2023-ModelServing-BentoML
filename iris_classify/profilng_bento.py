# iris_classify/profiling_bento.py
import numpy as np

from service import svc, Iris, IrisFeatures

runners = svc.runners

for runner in runners:
    runner.init_local(quiet=True)

sample_input = IrisFeatures(
    features=[
        Iris(
            sepal_len=6.2,
            sepal_width=3.2,
            petal_len=5.2,
            petal_width=2.2,
        )
        for _ in range(0, 1000) # 10 -> 1000
    ]
)

result: np.array = svc.apis["classify"].func(iris_features_pydantic=sample_input)

print(result)
