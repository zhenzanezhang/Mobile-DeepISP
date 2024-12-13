import torch
from model import MobileNetISP

def quantize_model(model_path='best_model.pth'):
    model = MobileNetISP(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
    )

    torch.save(quantized_model.state_dict(), 'quantized_model.pth')
    print("Quantized model saved.")

if __name__ == "__main__":
    quantize_model()
