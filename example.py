# Example file for training a LiESN-d with a single reservoir

import torch
from esn_dataloader import DataSplitter
from model.Metrics import index_agreement_torch
from model.ESN import LiESNd
import numpy as np

import faulthandler

if __name__ == "__main__":
    faulthandler.enable()

    # Input & target timeseries:
    #   "timeseries" : tuple of dicts, where each entry is one time series
    #   "n_folds" : optional parameter for using Kfold validation
    #   "train_batches" : optional parameter for using randomly split batches when training
    #   "val_batches" : optional parameter for using randomly split batches when training
    #   "device" : pytorch device where the data will be loaded
    #   "dtype" : data type that will be used for the dataset
    # The dict timeseries has the following parameters:
    #   "path" : relative path of the csv dataset
    #   "columns" : optional, name the columns that will be read
    #   "type" : descriptive type of the data, if columns parameter is missing, its the name of the column that will be read
    #   "is_predicted" : boolean that indicates to the dataloader if the time series will be predicted by the model
    #   "transformations" : list of transformations that can be applied to the data (eg: lowpass filtering, z-score)
    #   "description" : description of the data
    #   "allow_missing : boolean that indicates if gapes are allowed in the data when using random or sequential batching
    #   "missing_threshold" : indicates the maximum gap allowed for batching (in minutes) if "allow_missing" is True

    d = {'timeseries': (
    {'path': 'data/train/current_projection_praticagem.csv', 'type': 'velocity_projection',
     'is_input': True, 'is_predicted': True, 'transformations': [['low_pass_filter', 2]], 'description': 'ADCP',
     'allow_missing': True, 'missing_threshold': 60, 'blackout': True},
    {'path': 'data/train/current_sofs_praticagem_15min.csv', 'type': 'velocity_projection',
     'is_input': True, 'is_predicted': True, 'transformations': [], 'description': 'SOFS', 'allow_missing': True,
     'missing_threshold': 180, 'blackout': True}
    ),
    'device': torch.device('cuda'), 'dtype': torch.float32}

    # ESN parameters

    dict_train = {
        'spectral_radius': 0.95,
        'leak_rate': 0.35,
        'reservoir_size': 1000,
        'connectivity': 0.2,
        'input_scaling': 1.0,
        'bias_scaling': 1.0,
        'time_constant': 3600,
        'ridge_parameter': 0.005,
        "device": torch.device('cuda'),
        "torch_type": torch.float32,
        "loss": index_agreement_torch,
        "use_bias": True,
        "sparse": True,
        'seed': 24
    }

    dataloader = DataSplitter(**d)

    train_ds, validate_ds, dict_train['input_dim'], dict_train['output_dim'], dict_train[
        'output_map'], dict_train['input_map'] = dataloader.split_train_val(val_per=0.2,  # 20% data for validation
                                                                          single_training_batches=True, # Uses no batching for the training data
                                                                          batch_duration=240, # batch duration in hours
                                                                          sequential_validation=True, # Create sequential batches for validation
                                                                          sequential_stride=96, # The stride of validation batches in hours
                                                                          warmup=120 # ESN warmup
                                                                            )
    # ESN model (LiESNd)
    esn = LiESNd(**dict_train)
    # reset training state
    esn.reset()
    # train the esn in the training dataset
    esn.train_epoch(train_ds)
    # compute trained weights
    esn.train_finalize()

    inputs, predictions, losses = esn.predict_batches(validate_ds, forecast_horizon=120, warmup=120)

    print('Mean loss on validation (ADCP): ', np.mean(np.array(losses)[:,0]))

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