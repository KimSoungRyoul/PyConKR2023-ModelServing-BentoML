import numpy as np
import PIL.Image
import torch

import bentoml

pytorch_mnist_model = bentoml.pytorch.get("pytorch_mnist:4vivi3wlxgz64zx5")

runner = pytorch_mnist_model.to_runner()

runner.init_local()

img = PIL.Image.open("samples/0.png")
np_img = np.array(img)
tensor_img = torch.from_numpy(np_img).float()
tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
tensor_img = torch.nn.functional.interpolate(
    tensor_img, size=28, mode="bicubic", align_corners=False
)

# add color channel dimension for greyscale image
# arr = np.expand_dims(arr, 0)
result = runner.predict.run(tensor_img)  # => tensor(0)

print(result)
