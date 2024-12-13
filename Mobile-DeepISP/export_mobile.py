import torch
from model import MobileNetISP

def export_to_mobile(model_path='best_model.pth'):
    model = MobileNetISP(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 1, 256, 256)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save("model_mobile.pt")
    print("Mobile-compatible model saved as model_mobile.pt")

if __name__ == "__main__":
    export_to_mobile()
