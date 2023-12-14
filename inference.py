import torch
from sklearn.metrics import accuracy_score

try:
    from rich import print
except:
    pass

from params import training_params

device = training_params["device"]

def inference(model, test_dataloader):
    model.eval()
    state_dict = torch.load("output/model_weights.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    predictions, targets = [], []

    for batch, target in test_dataloader:
        batch = batch.to(torch.float32).to(device)
        target = target.type(torch.long).to(device)
        model.reset_state() # 模型维护了一种内部状态，这种状态如果不重置，会对模型的预测产生负面影响
        with torch.no_grad():
            out, _, _ = model(batch, record=True)
            peaks = torch.sum(out, dim=1)
            predictions.extend(peaks.argmax(1).cpu().numpy())
            targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(targets, predictions)
    print(f'Infer accuracy: {accuracy}')