import ttnn
import torch
import pytest
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.tt.ttnn_resnet import BasicBlock, ResNetFeatures


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [
        # (torch.rand((1, 64, 93, 306))), #layer1.0 #layer2.0
        # (torch.rand((1, 128, 47, 153))), #layer3.0
        # (torch.rand((1, 256, 24, 77))),
        # (torch.rand((1, 256, 24, 77))) #layer 4.0
        (torch.rand((1, 512, 12, 39)))
        # (torch.rand((1, 256, 159, 159))), #top down layer
    ],
    ids=["image"],
)
@pytest.mark.parametrize(
    "input_params",
    [
        # [[1, 93, 306, 64], [1, 93, 306, 64]], #layer1.0
        # [[1, 93, 306, 64], [1, 47, 153, 128]], #layer2.0
        # [[1, 47, 153, 128], [1, 24, 77, 256]], #layer3.0
        # [[1, 24, 77, 256], [1, 24, 77, 256]]
        # [[1, 24, 77, 256], [1, 12, 39, 512]] #layer 4.0
        [[1, 12, 39, 512], [1, 12, 39, 512]]
        # [[1, 159, 159, 256], [1, 159, 159, 256]], #layer4.0 shard is none
    ],
)
@pytest.mark.parametrize(
    "path",
    [
        # "frontend.layer1.0",
        # "frontend.layer2.0",
        # "frontend.layer3.1",
        "frontend.layer4.1",
        # "topdown.0",
    ],
)
def test_basic_block(device, input_tensor, input_params, path):
    disable_persistent_kernel_cache()

    torch.manual_seed(42)

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )

    state_dict = model.state_dict()

    model.load_state_dict(state_dict)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with torch.inference_mode():
        tt_module = BasicBlock(
            device,
            state_dict,
            path,
            input_params,
            inplanes=512,
            planes=512,
            stride=1,
        )
        ttnn_output = tt_module(device, ttnn_input)
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape(input_params[1])
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    torch_module = model.get_submodule(path)
    with torch.inference_mode():
        torch_output = torch_module(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [(torch.rand((1, 3, 370, 1224)))],
    ids=["image"],
)
def test_resnet_features(device, input_tensor):
    disable_persistent_kernel_cache()

    torch.manual_seed(42)

    model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=8,
        grid_res=0.5,
        grid_height=4.0,
    )

    state_dict = model.state_dict()

    model.load_state_dict(state_dict)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    torch_module = model.get_submodule("frontend")
    with torch.inference_mode():
        torch_output = torch_module(input_tensor)[2]

    with torch.inference_mode():
        tt_module = ResNetFeatures(
            device,
            state_dict,
            "frontend",
            BasicBlock,
            [2, 2, 2, 2],
        )
        ttnn_output = tt_module(device, ttnn_input)[2]
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.permute((0, 3, 1, 2))

    passing, pcc = assert_with_pcc(ttnn_output, torch_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")
