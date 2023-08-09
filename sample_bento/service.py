import asyncio

import bentoml
import numpy as np
import torch
from torch import Tensor

#from middleware import CustomHeaderMiddleware



runner = bentoml.pytorch.get("sample-dummy-model:latest").to_runner()

svc = bentoml.Service(name="sample-dummy-bento", runners=[runner])
#svc.add_asgi_middleware(CustomHeaderMiddleware, path_mapping=["predict",])


#@profile
@svc.api(
    input=bentoml.io.NumpyNdarray.from_sample(np.array([[1.1, 2.2], [3.3, 4.4]])),
    output=bentoml.io.NumpyNdarray.from_sample(np.array([[0.0, 0.0], [0.0, 0.0]])),
    doc="description....",
)
def predict(input_array: np.array) -> np.array:
    # asyncio.sleep(1)
    inference_output: Tensor = runner.run(torch.tensor(input_array))
    return inference_output.detach().cpu().numpy()
