import wandb
from torch.utils.data import DataLoader, random_split
from src import load_and_preprocess_data, ECGDataset, train_snn_model
from utils import dataset_params, training_params, SweepConfig
from models import MyNet

if __name__ == '__main__':

    model = MyNet

    sweep_id = wandb.sweep(SweepConfig, project="snn-ecg_0329", entity="ykzhang2023")

    x_data, y_data = load_and_preprocess_data()
    
    def train():
        with wandb.init() as run:
            config = run.config

            dataset = ECGDataset(x_data, y_data, training_params["Num_Datas"], config.time_partitions, config.voltage_partitions)
            train_size = int((1 - dataset_params["RATIO"]) * len(dataset))
            test_size = len(dataset) - train_size

            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
            
            train_snn_model(model, train_dataloader, test_dataloader, config)

            # from tests.test_xylosim import xyloSim_inference
            # xylosim_infer_acc = xyloSim_inference(test_dataloader)
            # print(f'XyloSim Inference Accuracy: {xylosim_infer_acc:.4f}')

    wandb.agent(sweep_id, train, count=10)

    