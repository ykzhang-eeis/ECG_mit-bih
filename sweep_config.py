import math

sweepConfig = {
                'method': 'grid',
                'metric': {'goal': 'minimize', 'name': 'loss'},
                'parameters': {
                    'batch_size': {'values': [32, 64, 128]},
                    'time_partitions': {'values': [15, 30, 60, 75]},
                    'voltage_partitions': {'values': [16, 32, 64, 80]},
                    'num_epochs': {'value': 100},
                    'learning_rate': {'values': [1e-3, 5e-4, 1e-4]},
                    'optimizer': {'values': ['adam', 'sgd']}
                }
 }