from time import sleep

import bentoml
import pytest
import torch

from service import svc
from numpy import array
#bento = bentoml.get("sample-dummy-bento:latest")

for runner in svc.runners:
    runner.init_local(quiet=True)

#inference_input = torch.tensor([[1.1, 2.2], [3.3, 4.4]], dtype=torch.float32)

#
# if __name__ == '__main__':
#     for _ in range(0,100_000):
#         response = svc.apis["predict"].func(array([[1.1, 2.2], [3.3, 4.4]]))
#     #print(response)

if __name__ == '__main__':
    #bento = bentoml.get("sample-dummy-bento:ksr-temp-kkk")

    server = bentoml.HTTPServer("service.py:svc",production=False, port=3033)

    server.start(text=True, blocking=False)


    sleep(10)
    client = server.get_client()

    response = client.predict(array([[1.1, 2.2], [3.3, 4.4]]))
    print(response)
    server.stop()