import torch
from numpy import array

from service import svc

for runner in svc.runners:
    runner.init_local(quiet=True)

inference_input = torch.tensor([[1.1, 2.2], [3.3, 4.4]], dtype=torch.float32)


if __name__ == '__main__':
    for _ in range(0,100_000):
        response = svc.apis["predict"].func(array([[1.1, 2.2], [3.3, 4.4]]))
    print(response)
