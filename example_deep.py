# Example file for training a LiESN-d with a single reservoir

import torch
from esn_dataloader import DataSplitter
from model.Metrics import index_agreement_torch
from model.ESN import DeepLiESNd
import numpy as np

if __name__ == "__main__":
    # Input & target timeseries

    d = {'timeseries': (
    {'path': 'data/train/current_projection_praticagem.csv', 'type': 'velocity_projection',
     'is_input': True, 'is_predicted': True, 'transformations': [['low_pass_filter', 2]], 'description': 'ADCP',
     'allow_missing': True, 'missing_threshold': 60, 'blackout': True},
    {'path': 'data/train/current_sofs_praticagem_15min.csv', 'type': 'velocity_projection',
     'is_input': True, 'is_predicted': True, 'transformations': [], 'description': 'SOFS', 'allow_missing': False,
     'missing_threshold': 180, 'blackout': True}
    ),
    'device': torch.device('cuda'), 'dtype': torch.float32}

    # ESN parameters
    # First layer reservoirs names should be the same of the time series description. This will be used to map the time
    # series to the correct reservoirs

    dict_train = {"layers": {
        "Layer1": {"ADCP": {
            'spectral_radius': 0.9,
            'leak_rate': 0.5,
            'reservoir_size': 500,
            'connectivity': 0.2,
            'input_scaling': 1,
            'bias_scaling': 1,
            'time_constant': 3600,
            'ridge_parameter': 0.0000,
            'input_dim': 1,
            'output_dim': 1
        },
            "SOFS": {
                'spectral_radius': 0.9,
                'leak_rate': 0.5,
                'reservoir_size': 500,
                'connectivity': 0.2,
                'input_scaling': 1,
                'bias_scaling': 1,
                'time_constant': 3600,
                'ridge_parameter': 0.0000,
                'input_dim': 1,
                'output_dim': 1
            },
        },
        "Layer2": {
            "DeepLayer1": {
                'spectral_radius': 0.5,
                'leak_rate': 0.2,
                'reservoir_size': 500,
                'connectivity': 0.6,
                'input_scaling': 1,
                'bias_scaling': 1,
                'time_constant': 3600,
                'ridge_parameter': 0.0000,
                'input_dim': 500 + 500,
                'output_dim': 1
            },
            "DeepLayer2": {
                'spectral_radius': 0.5,
                'leak_rate': 0.2,
                'reservoir_size': 500,
                'connectivity': 0.6,
                'input_scaling': 1,
                'bias_scaling': 1,
                'time_constant': 3600,
                'ridge_parameter': 0.0000,
                'input_dim': 500 + 500,
                'output_dim': 1
            }
        },
        "Layer3": {"Joiner": {
            'spectral_radius': 0.9,
            'leak_rate': 0.5,
            'reservoir_size': 500,
            'connectivity': 0.2,
            'input_scaling': 1,
            'bias_scaling': 1,
            'time_constant': 3600,
            'ridge_parameter': 0.0000,
            'input_dim': 500 + 500,
            'output_dim': 1
        },
        },
    },
        "device": torch.device('cuda'),
        "torch_type": torch.float32,
        "loss": index_agreement_torch,
        "input_dim": 2,
        "use_bias": True,
        "sparse": True,
        "ridge": 0.01,
        "time_constant": 3600,
        'seed': 38,
        "output_map": None
    }

    dataloader = DataSplitter(**d)

    train_ds, validate_ds, dict_train['input_dim'], dict_train['output_dim'], dict_train[
        'out_maps'], dict_train['input_map'] = dataloader.split_train_val(val_per=0.2,  # 20% data for validation
                                                                            single_training_batches=True,
                                                                            # Uses no batching for the training data
                                                                            batch_duration=240,
                                                                            # batch duration in hours
                                                                            sequential_validation=True,
                                                                            # Create sequential batches for validation
                                                                            sequential_stride=96,
                                                                            # The stride of validation batches in hours
                                                                            warmup=120  # ESN warmup
                                                                            )
    # ESN model (MTCLiESN = LiESNd)
    esn = DeepLiESNd(**dict_train)
    # reset training state
    esn.reset()
    # train the esn in the training dataset
    esn.train_epoch(train_ds)
    # compute trained weights
    esn.train_finalize()

    inputs, predictions, losses = esn.predict_batches(validate_ds, forecast_horizon=120, warmup=120)

    print('Mean loss on validation (ADCP): ', np.mean(np.array(losses)[:, 0]))

    # Now we will evaluate the test dataset

    d_test = {'timeseries': (
    {'path': 'data/test/current_projection_praticagem.csv', 'type': 'velocity_projection',
     'is_input': True, 'is_predicted': True, 'transformations': [['low_pass_filter', 2]], 'description': 'ADCP',
     'allow_missing': True, 'missing_threshold': 60, 'blackout': True},
    {'path': 'data/test/current_sofs_praticagem_15min.csv', 'type': 'velocity_projection',
     'is_input': True, 'is_predicted': True, 'transformations': [], 'description': 'SOFS', 'allow_missing': True,
     'missing_threshold': 180, 'blackout': True}
    ),
    'device': torch.device('cuda'), 'dtype': torch.float32}

    test_dl = DataSplitter(**d_test)

    # Two datasets, one with blackout before the forecast and a second without blackout

    test_missing= test_dl.sequential_batches_missing_before_forecast(batch_duration=240,
                                                                  stride=96,
                                                                  forecast_window=120,
                                                                  missing_hours=60)

    test_no_missing= test_dl.sequential_batchs(batch_duration=240,stride=96)

    inputs, predictions, losses = esn.predict_batches(test_no_missing, forecast_horizon=120, warmup=120)

    print('Mean loss on test without blackout (ADCP): ', np.mean(np.array(losses)[:,0]))

    inputs, predictions, losses = esn.predict_batches(test_missing, forecast_horizon=120, warmup=120)

    print('Mean loss on validation w/ 60 hour blackout (ADCP): ', np.mean(np.array(losses)[:,0]))