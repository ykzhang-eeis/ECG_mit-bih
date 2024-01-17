import os
import wandb
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from rich import print
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
from tqdm import tqdm


from params import training_params

import warnings
warnings.filterwarnings('ignore')

def train_snn_model(model, train_dataloader, test_dataloader, config=None) -> None:
    device = training_params["device"]
    lr = config.learning_rate if config else training_params["Learning_Rate"]
    epochs = config.num_epochs if config else training_params["Num_Epochs"]

    model.to(device)
    criterion = CrossEntropyLoss()
    wandb.watch(model, criterion, log="all", log_freq=10)
    optimizer = Adam(model.parameters().astorch(), lr=lr)

    best_val_f1 = 0.0
    for epoch in range(epochs):
        train_loss, train_f1 = run_epoch(train_dataloader, model, criterion, optimizer, device, train=True)
        test_metrics = run_epoch(test_dataloader, model, criterion, optimizer, device, train=False)
        test_loss, test_f1, test_precision, test_recall, test_accuracy = test_metrics[:5]

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Train F1: {train_f1}")
        print(f"Test Loss: {test_loss}, Test F1: {test_f1}")
        print(f"Precision: {test_precision}, Recall: {test_recall}, ACC: {test_accuracy}")
        wandb.log({'epoch': epoch+1, 'loss': train_loss, 
                   'f1': train_f1, 'test_loss': test_loss, 
                   'test_f1': test_f1, 'test_precision': test_precision, 
                   'test_recall': test_recall, 'test_accuracy': test_accuracy})

        test_predictions, test_targets = test_metrics[5:]
        print(confusion_matrix(test_targets, test_predictions))

        if test_f1 > best_val_f1:
            best_val_f1 = test_f1
            if not os.path.exists("output"):
                os.makedirs("output")
            torch.save(model.state_dict(), "output/model_weights.pth")

def run_epoch(dataloader, model, criterion, optimizer, device, train):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    predictions, targets = [], []

    for batch, target in tqdm(dataloader):
        batch = batch.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.long)

        model.reset_state()
        optimizer.zero_grad() if train else None
        
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
        accuracy = accuracy_score(targets, predictions)
        precision, recall, _, _ = precision_recall_fscore_support(targets, predictions, average='macro')
        return total_loss, f1, precision, recall, accuracy, predictions, targets
    
    return total_loss, f1