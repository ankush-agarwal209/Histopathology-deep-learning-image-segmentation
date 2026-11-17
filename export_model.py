
import torch
from model import UNet

model = UNet()
checkpoint = torch.load(
    "outputs/best_unet.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, 3, 512, 512)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["segmentation_logits"],
    opset_version=17,
    dynamo=False
)

print("Model exported successfully!")
