import torch
import numpy as np
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

def model_train_snn(train_dataloader, test_dataloader, model=SynNet(4,15)):
    model.to(training_params["device"])
    class_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0]).to(training_params["device"])
    criterion = CrossEntropyLoss(weight=class_weights)
    # criterion = CrossEntropyLoss()
    opt = Adam(model.parameters().astorch(), lr=training_params["Learning_Rate"])
    best_val_f1 = 0
    for epoch in range(training_params["Num_Epochs"]):
        train_preds = []
        train_targets = []
        sum_loss = 0.0
        for batch, target in tqdm.tqdm(train_dataloader):
            batch = batch.to(torch.float32).to(training_params["device"])
            target = target.type(torch.LongTensor).to(training_params["device"])
            model.reset_state()
            opt.zero_grad()
            out, _,rec = model(batch, record = True)
            # peaks = out.max(1)[0].to(training_params["device"])
            peaks = torch.sum(out,dim=1)
            # peaks = out.squeeze(0)
            # print(peaks.argmax(1)==target)
            loss = criterion(peaks, target)
            l2_reg = 0.0
            for param in model.parameters():
                if(type(param) == str):
                    continue
                else:
                    l2_reg += torch.norm(param)**2
            loss += training_params["lambda_reg"] * l2_reg
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = peaks.argmax(1).detach()
                train_preds += pred.detach().cpu().numpy().tolist()
                train_targets += target.detach().cpu().numpy().tolist()
                sum_loss += loss.item()/training_params["Num_Datas"]
        sum_f1 = f1_score(train_targets, train_preds, average="macro") # 输出的是所有分类的f1-score
        print(
            f"Train Epoch = {epoch+1}, Loss = {sum_loss}, sum F1 Score = {sum_f1}"
        )
        test_preds = []
        test_targets = []
        test_loss = 0.0
        for batch, target in tqdm.tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.to(torch.float32).to(training_params["device"])
                target = target.type(torch.LongTensor).to(training_params["device"])
                model.reset_state()
                out, _,rec = model(batch, record = True)
                peaks = torch.sum(out,dim=1)
                # peaks = out.squeeze(0)
                # print(peaks.argmax(1)==target)
                pred = peaks.argmax(1).detach().to(training_params["device"])
                loss = criterion(peaks, target)
                test_loss += loss.item()/training_params["Num_Datas"]
                test_preds += pred.detach().cpu().numpy().tolist()
                test_targets += target.detach().cpu().numpy().tolist()
        f1 = f1_score(test_targets, test_preds, average="macro")
        print(confusion_matrix(test_targets, test_preds))
        test_p, test_r,_,_ =  precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(dataset_params["CLASSES"])
        )
        print(f"Val Loss = {test_loss}, Val Precision = {test_p}, Recall = {test_r}")
        print(f"Val Epoch = {epoch+1}, bestf1score = {best_val_f1}, f1score = {f1}")
        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), "output/model_weights.pth")
            # model.save("output/model_best.json")