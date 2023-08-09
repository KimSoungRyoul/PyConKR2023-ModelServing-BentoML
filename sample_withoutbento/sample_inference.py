import torch
from torch import Tensor


# sample inference pytorch model
# 1. define inference Model
class SampleDummyModel(torch.nn.Module):
    def forward(self, x_tensor) -> Tensor:
        # pytorch Module, only return transposed tensor
        return torch.transpose(x_tensor, 0, 1)


# 2. load model
model = SampleDummyModel()
model.load_state_dict(torch.load("./sample_dummy_model.pt"))
model.eval()

sample_input = torch.tensor([[1.1, 2.2], [3.3, 4.4]], dtype=torch.float32)

# 3. inference
inference_output = model(sample_input)

print(inference_output)
