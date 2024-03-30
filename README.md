#  ECG Classification with SNN on Xylo Audio v2

## Overview

This project leverages the power of Rockpool, a technology from Synsense, to build a Spiking Neural Network (SNN) aimed at solving a four-category classification problem in Electrocardiogram (ECG) analysis. By harnessing the efficiency and dynamics of SNNs, we've managed to achieve an F1-score of 0.95, demonstrating high accuracy in distinguishing between the four ECG categories. Furthermore, this project has been successfully deployed on the Xylo Audio v2 development board, showcasing its real-world applicability.

## Key Features

- **High Performance**: Achieved an F1-score of 0.95 in ECG classification, indicating excellent precision and recall in identifying the four distinct categories within ECG signals.
- **Deployment Ready**: The project has been fully deployed on the Xylo Audio v2 development board, highlighting its practicality for use in medical devices and health monitoring systems.
- **Efficiency**: Every neuron in the network operates with a power consumption of less than 0.2mW, making it extremely energy-efficient and suitable for long-term, continuous monitoring applications.

## Technologies Used

- [**Rockpool**](https://rockpool.ai/): A framework developed by Synsense, tailored for building and simulating Spiking Neural Networks (SNNs). Rockpool's efficiency and flexibility are critical in modeling complex neural dynamics required for high-accuracy ECG classification.
- [**Xylo Audio v2 Development Board**](https://www.synsense.ai/wp-content/uploads/2023/06/Xylo-Audio-datasheet.pdf): Chosen for its low power consumption and robust processing capabilities, making it an ideal platform for deploying energy-efficient neural networks.

## Getting Started

To replicate this project or explore its capabilities, ensure you have access to the Rockpool framework and a Xylo Audio v2 development board. Due to the specific nature of the hardware and software used, familiarity with SNNs and experience in embedded systems programming will be beneficial.

### Step 1: Install Required Python Packages

Clone the project repository to your local machine, then navigate to the project directory in your terminal or command prompt. Install the required Python packages by running the following command:

```shell
pip install -r requirements.txt
```

This command reads the `requirements.txt` file in the project directory and installs all the dependencies listed there.

###  Step 2: Download the MIT-BIH Arrhythmia Database

The next step involves downloading the MIT-BIH Arrhythmia Database, which is used as the dataset for training our model. Please follow these steps:

1. Go to the [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) website or the specific page provided by the project documentation to access the MIT-BIH Arrhythmia Database.
2. Download the dataset to your local machine.
3. Extract the dataset and move it to the `Dataset` directory within your project folder. If the `Dataset` directory does not exist, please create it at the root of the project directory.

Ensure that the dataset files are correctly placed so that the script can access them without any issues.

### Step 3: Start the Training Process

Once you have installed all required packages and prepared the dataset, you can start the training process by running:

```shell
python main.py
```

### Connecting to Weights & Biases (Wandb)

This project uses Weights & Biases (Wandb) for hyperparameter tuning and visualizing metrics. To connect your training process to Wandb, follow these steps:

1. If you haven't already, sign up for a Wandb account at https://wandb.ai/.
2. Install the Wandb package (if it's not already included in the `requirements.txt` file) by running `pip install wandb`.
3. Log in to Wandb by executing `wandb login` in your terminal and following the instructions to authenticate.
4. Modify the `main.py` script to include your Wandb project name and any other relevant configurations for logging.

The script will automatically log metrics to your Wandb dashboard, allowing you to monitor the training process, visualize performance metrics, and manage hyperparameter searches.