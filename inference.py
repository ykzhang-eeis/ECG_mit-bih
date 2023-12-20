import torch
from rich import print
from sklearn.metrics import accuracy_score

from params import training_params


device = training_params["device"]

def inference(model, test_dataloader) -> float:
    state_dict = torch.load("output/model_weights.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    predictions, targets = [], []
    with torch.no_grad():  # Disable gradient computation
        for batch, target in test_dataloader:
            batch = batch.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)
            model.reset_state()  # Reset internal state of the model

            out, _, _ = model(batch, record=True)
            peaks = torch.sum(out, dim=1)
            predictions.extend(peaks.argmax(1).cpu().numpy())
            targets.extend(target.cpu().numpy())

    return accuracy_score(targets, predictions)