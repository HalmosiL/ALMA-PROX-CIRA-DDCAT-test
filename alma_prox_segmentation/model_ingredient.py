from model import load_model, get_new_model

def requires_grad_(model, requires_grad):
    for param in model.parameters():
        param.requires_grad_(requires_grad)

def get_model(device, path):
    model = load_model(path, device=device)
    model.eval()
    model.to(device)
    requires_grad_(model, False)
    return model