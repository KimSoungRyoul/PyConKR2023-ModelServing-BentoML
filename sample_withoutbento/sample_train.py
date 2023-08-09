import bentoml.torchscript
import torch
from torch import Tensor


# sample train pytorch model
# 1. define Model
class SampleDummyModel(torch.nn.Module):
    def forward(self, x_tensor) -> Tensor:
        # pytorch Module, only return transposed tensor
        return torch.transpose(x_tensor, 0, 1)


model = SampleDummyModel()

# 2. something model training code ....

# 3. finally save trained model to file
torch.save(model.state_dict(), "./sample_dummy_model.pt")
#
# bentoml.pytorch.save_model(
#     name="sample-dummy-model:2023-08-12", # {model-name}:{model-version}
#     model,
# )
