import torch
import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from rockpool.nn.networks import SynNet
from params import training_params, dataset_params

try:
    from rich import print
except:
    pass

import warnings
warnings.filterwarnings('ignore')

classes = dataset_params["CLASSES"]
channels = dataset_params["Time_Partitions"]

def model_train_snn(train_dataloader, test_dataloader, model=SynNet(classes, channels)):
    device = training_params["device"]
    lr = training_params["Learning_Rate"]
    epochs = training_params["Num_Epochs"]
    model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters().astorch(), lr=lr)
    best_val_f1 = 0
    for epoch in range(epochs):
        train_loss, train_f1 = run_epoch(train_dataloader, model, criterion, optimizer, device, train=True)
        test_loss, test_f1, test_precision, test_recall, test_predictions, test_targets = run_epoch(test_dataloader, model, criterion, optimizer, device, train=False)
        test_confusion_matrix = confusion_matrix(test_targets, test_predictions)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Train F1: {train_f1}")
        print(f"Test Loss: {test_loss}, Test F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall}")
        print(f"Test Confusion Matrix:\n{test_confusion_matrix}")
        if test_f1 > best_val_f1:
            best_val_f1 = test_f1
            torch.save(model.state_dict(), "output/model_weights.pth")

def run_epoch(dataloader, model, criterion, optimizer, device, train):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    predictions, targets = [], []
    for batch, target in tqdm.tqdm(dataloader):
        batch = batch.to(torch.float32).to(device)
        target = target.type(torch.long).to(device)

        model.reset_state()
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            outputs, _, _ = model(batch, record=True)
            peaks = torch.sum(outputs, dim=1)
            loss = criterion(peaks, target)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() / len(dataloader.dataset)
        predictions.extend(peaks.argmax(1).detach().cpu().numpy())
        targets.extend(target.detach().cpu().numpy())

    f1 = f1_score(targets, predictions, average="macro")
    if not train:
        precision, recall, _, _ = precision_recall_fscore_support(targets, predictions, average='macro')
        return total_loss, f1, precision, recall, predictions, targets
    return total_loss, f1