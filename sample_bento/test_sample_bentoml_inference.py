import bentoml
import pytest
import torch


@pytest.mark.parametrize(
    argnames="sample_input,expected_output",
    argvalues=[
        (
            # sample_input
            [[1.1, 2.2], [3.3, 4.4]],
            # expected_output
            [[1.1, 3.3], [2.2, 4.4]],
        ),
    ],
)
def test_model_inference(sample_input, expected_output):
    # 1. get(load) model & runner
    # model = SampleDummyModel()
    # model.load_state_dict(torch.load("./sample_dummy_model.pt"))
    # model.eval()

    # s3 download bentoModel
    bentoml.models.import_model(
        "s3://pycon-sample-s3/bento-models-folder/sample-dummy-model:2023-08-13.bentomodel",
        user="<AWS access key>", passwd="<AWS secret key>",
    )

    sample_dummy_model = bentoml.pytorch.get("sample-dummy-model:latest")
    runner = sample_dummy_model.to_runner()
    runner.init_local(quiet=True)

    inference_input = torch.tensor(sample_input, dtype=torch.float32)

    # 3. inference
    # inference_output = model(sample_input)
    inference_output = runner.run(inference_input)

    print(inference_output)
    assert (
        inference_output.equal(torch.tensor(expected_output, dtype=torch.float32))
        is True
    )
