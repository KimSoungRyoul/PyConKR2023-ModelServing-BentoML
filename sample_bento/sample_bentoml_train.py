from datetime import date, datetime

import bentoml
import torch
from torch import Tensor
from bentoml import Model


# sample train pytorch model
# 1. define Model
class SampleDummyModel(torch.nn.Module):
    def forward(self, x_tensor) -> Tensor:
        # pytorch Module, only return transposed tensor
        return torch.transpose(x_tensor, 0, 1)


model = SampleDummyModel()

# 2. something model training code ....

# 3. finally save trained model to file
# torch.save(model.state_dict(), "./sample_dummy_model.pt")
bento_model: Model = bentoml.pytorch.save_model(
    name=f"sample-dummy-model:{datetime.today().strftime('%Y-%m-%d')}",  # {model-name}:{model-version}
    model=model,
    labels={  # you can use labeling which managed by BentoML
        "maintainer": "KimSoungRyoul",
    },
)

bentoml.models.export_model(
    tag=bento_model.tag,
    path="s3://pycon-sample-s3/bento-models-folder/sample-dummy-model:2023-08-13.bentomodel",
    user="<AWS access key>", passwd="<AWS secret key>",
)



print(f"name:{bento_model.info.name}")  # sample-dummy-model:20230701181120140830
print(f"tag:{bento_model.info.tag}")  # sample-dummy-model:20230701181120140830
print(f"version: {bento_model.info.version}")  # '20230701181120140830'
print(f"labels:{bento_model.info.labels}")  # {'maintainer': 'KimSoungRyoul'}


# show sample-dummy-model List
bento_models = bentoml.models.list(
    "sample-dummy-model"
)  # $ bentoml models list sample-dummy-model
for bento_model in bento_models:
    print(bento_model)  # Model(tag="sample-dummy-model:20230701181548369470"
