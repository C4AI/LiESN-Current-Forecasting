import os

#from Model.ESN_sofs_operator_missing import MTCLiESN
from model.ESN import LiESNd
from model.Metrics import index_agreement_torch
from esn_dataloader import DataSplitter
import torch
import matplotlib.pyplot as plt
import sys, traceback
import time
import wandb
import numpy as np
import pandas as pd
import pickle
import json

def train_deepesn(steps):
    wandb.login(key=train_parameters["wandb_api_key"])     # log into Weights and Biases

    # Configure Sweep hyperparameter ranges and search algorithm in a dict

    sweep_config = {
        'method': 'bayes',
        'name':'sweep',
        'metric': {
            'name': 'ioa_val',
            'goal': 'maximize'
        },
        'parameters': {
            #ADCP
            'spectral_radius': {'max': 1.0, 'min': 0.0001},
            'leak_rate': {'max': 1.0, 'min': 0.0000},
            'reservoir_size': {'values': [x for x in range(300, 7100, 100)]},
            'connectivity': {'max': 0.8, 'min': 0.0001},
            'input_scaling': {'max': 5.0, 'min': 0.0001},
            'bias_scaling': {'max': 5.0, 'min': 0.0000},
            'time_constant': {'max': 1800, 'min': 30},
            'ridge_parameter': {'max': 0.3, 'min': 0.0000001},
        }
    }

    try:
        if train_parameters['Sweep_id'] is None:
            train_parameters['Sweep_id'] = wandb.sweep(sweep_config, project=train_parameters['Project_name'], entity=train_parameters['username'])

        # uncomment the line below to join an already existing sweep
        wandb.agent(sweep_id=train_parameters['Sweep_id'], project=train_parameters['Project_name'], function=DPESN_train, count=steps)

    except Exception as e:
    # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)

def config_2_dict(**config):

    dict_train = {
            'spectral_radius': config['spectral_radius'],
            'leak_rate': config['leak_rate'],
            'reservoir_size': config['reservoir_size'],
            'connectivity': config['connectivity'],
            'input_scaling': config['input_scaling'],
            'bias_scaling': config['bias_scaling'],
            'time_constant': config['time_constant'],
            'ridge_parameter': config['ridge_parameter'],
            'input_dim': 1,
            'output_dim': 1,
            "device": torch.device('cuda'),
            "torch_type": torch.float32,
            "loss": index_agreement_torch,
            "use_bias": True,
            "sparse": True,
            "ridge": config["ridge_parameter"],
            'seed': train_parameters['seed'],
            "output_map": None
    }
    return dict_train

# Training function, warmup is the esn warmup in hours and the forecasting horizon is how my hours will be predicted into the future

def DPESN_train(config=None):

    try:

        with wandb.init(config=config, project=train_parameters['Project_name'], entity=train_parameters['username']):
            api = wandb.Api()
            sp = api.sweep(train_parameters['username'] + '/' + train_parameters['Project_name'] + '/' + train_parameters['Sweep_id'])  # Recover sweep data

            # Find best IoA in the sweep
            br = sp.best_run()
            if br is not None:
                sbr = br.summary
                if sbr is not None:
                    if 'ioa_val' in sp.best_run().summary.keys():
                        best_ioa = sp.best_run().summary['ioa_val']     # Best current IoA to be beaten
                    else:
                        best_ioa = 0.0
                else:
                    best_ioa = 0.0
            else:
                best_ioa = 0.0

            config = wandb.config       # Receives the hyperparameter dict from wandb
            dict_config = dict(config)
            dict_train = config_2_dict(**dict_config)       # Builds the Deep ESN dictionary
            train_dl = DataSplitter(**d)       # Dataloader

            # Split data into training and validation
            train_ds, validate_ds, dict_train['input_dim'], dict_train['output_dim'], dict_train[
                'output_map'], dict_train['input_map'] = train_dl.split_train_val(val_per=0.2,        # 20% data for validation
                                                                                single_training_batches=True,   # Uses no batching for the training data
                                                                                batch_duration=train_parameters["warmup"] + train_parameters["forecast_horizon"],    # batch duration in hours
                                                                                sequential_validation=True,     # Create sequential batches for validation
                                                                                sequential_stride=train_parameters["validation_stride"],    # The stride of validation batches in hours
                                                                                warmup=pd.Timedelta(7, unit='days'))
            start = time.time()
            # create Time Continous ESN
            esn = LiESNd(**dict_train)
            # reset training state
            esn.reset()
            # train the esn in the training dataset
            esn.train_epoch(train_ds)
            # compute trained weights
            esn.train_finalize()
            # Calculate loss in the validation dataset
            inputs, predictions, losses = esn.predict_batches(validate_ds, forecast_horizon=train_parameters["forecast_horizon"], warmup=train_parameters["warmup"])
            losses = np.array(losses)
            fold_losses = np.nanmean(losses)
            fold_std = np.nanstd(losses)

            mean_loss = np.mean(fold_losses)
            mean_std = np.mean(fold_std)
            # Log metrics into the wandb portal
            wandb.log({"ioa_val": mean_loss, "std_ioa_val": mean_std})

            # if training surpassed the previous best run, we save the weights
            if mean_loss > best_ioa:
                esn.save_weights(train_parameters['save_path'] + train_parameters['save_name'])

            end = time.time()
            print("Finished one run! Total time: " + str(end-start))
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)


def evaluate():

    # Recover trained parameters
    try:
        dict_config = json.load(open(train_parameters['save_path'] + "config_dict.json", 'r'))
    except FileNotFoundError:
        wandb.login(key=train_parameters["wandb_api_key"])
        api = wandb.Api()
        wandb.init(project=train_parameters['Project_name'], entity=train_parameters['username'])
        runs = api.sweep(train_parameters['username'] + '/' + train_parameters['Project_name'] + '/' + train_parameters['Sweep_id'])    # Run to be evaluated
        run = runs.best_run()
        # Recover hyperparameters from the wandb and create the ESN dataset
        config = run.config
        dict_config = dict(config)
        json.dump(dict_config, open(train_parameters['save_path'] + "config_dict.json", "w"))

    dict_train = config_2_dict(**dict_config)

    # Trainign dataset
    train_dl = DataSplitter(**d)
    train_ds, validate_ds, dict_train['input_dim'], dict_train['output_dim'], dict_train[
        'output_map'], dict_train['input_map'] = train_dl.split_train_val(val_per=0.0,
                                                                        single_training_batches=True,
                                                                        batch_duration=train_parameters["forecast_horizon"] + train_parameters["warmup"],
                                                                        sequential_validation=True,
                                                                        sequential_stride=train_parameters["validation_stride"])

    dict_train['input_dim'] = 1

    # Test dataset
    test_dl = DataSplitter(**d_test)

    if 'missing_hours' in train_parameters.keys():
        test = test_dl.sequential_batches_missing_before_forecast(batch_duration=train_parameters["forecast_horizon"] + train_parameters["warmup"], stride=train_parameters["forecast_horizon"], forecast_window=train_parameters['forecast_horizon'],missing_hours=train_parameters['missing_hours'])
    else:
        test = test_dl.sequential_batchs(batch_duration=train_parameters["forecast_horizon"] + train_parameters["warmup"],
                                        stride=train_parameters["forecast_horizon"])

    # Create deep Time continous ESN
    esn = LiESNd(**dict_train)
    #esn.load_weights(train_parameters['save_path'] + train_parameters['save_name'] + '_fulltrained')
    # Optional: Retrain the ESN with all the train/validate dataset
    try:
        esn.load_weights(train_parameters['save_path'] + train_parameters['save_name'] + '_fulltrained')
        print('loaded fully trained weights')
    except FileNotFoundError:
        try:
            # Load weights trained with 80% of the data
            esn.load_weights(train_parameters['save_path'] + train_parameters['save_name'])
            print('loaded trained weights')
        except FileNotFoundError:
            pass
        esn.reset()
        esn.train_epoch(train_ds)
        esn.train_finalize()
        esn.save_weights(train_parameters['save_path'] + train_parameters['save_name']+ '_fulltrained')

    # Predict using the test dataset
    inputs, predictions, losses = esn.predict_batches(test, forecast_horizon=train_parameters["forecast_horizon"], warmup=train_parameters["warmup"])


    losses_test = np.array(losses)
    print('mean losses test:' + str(np.mean(losses_test[:,0])) + '+-' + str(np.std(losses_test[:,0])))

    path = train_parameters['save_path'] + train_parameters['save_path_output'] + '/'
    os.makedirs(path, exist_ok=True)

    if 'missing_hours' in train_parameters.keys():
        with open(path + str(int(train_parameters["missing_hours"])) + 'gap_hours_missing.pickle', 'wb') as handle:
            dta = inputs, predictions, losses
            pickle.dump(dta, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(train_parameters["save_path"] + 'no_gap.pickle', 'wb') as handle:
            dta = inputs, predictions, losses
            pickle.dump(dta, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    # train/validate dict
    d = {'timeseries': ({'path': 'data/train/current_projection_praticagem.csv', 'type': 'velocity_projection', 'is_input': False, 'is_predicted': True, 'transformations': [['low_pass_filter', 2]], 'description': 'ADCP', 'allow_missing':True, 'missing_threshold': 60, 'is_forecast': False},
                        {'path': 'data/train/current_sofs_praticagem_15min.csv', 'type': 'velocity_projection', 'is_input': True, 'is_predicted': True, 'transformations': [], 'description': 'SOFS', 'allow_missing':False, 'missing_threshold': 180, 'is_forecast': False}),
         'device': torch.device('cuda'), 'dtype': torch.float32}


    # test dict
    d_test = {'timeseries': ({'path': 'data/test/current_projection_praticagem.csv', 'type': 'velocity_projection', 'is_input': False, 'is_predicted':True,'transformations': [['low_pass_filter', 2]], 'description': 'ADCP', 'allow_missing':True, 'missing_threshold': 60,  'is_forecast': False},
                             {'path': 'data/test/current_sofs_praticagem_15min.csv', 'type': 'velocity_projection', 'is_input': True, 'is_predicted':True,'transformations': [], 'description': 'SOFS', 'allow_missing':False, 'missing_threshold': 180, 'is_forecast': False}),
         'device': torch.device('cuda'), 'dtype': torch.float32}

    # path where the weights will be saved locally

    train_parameters = json.load(open("esn_config_architecture_B.json", "r"))

    os.makedirs(train_parameters["save_path"], exist_ok=True)

    if all([train_parameters['wandb_api_key'], train_parameters['username'], train_parameters['Project_name']]):
        train_deepesn(train_parameters['steps'])

    blackout = [24, 48, 72, 96, 120, 144]
    evaluate()

    for m in blackout:
        train_parameters['save_path_output'] = str(int(m)) + '_hours_gap'
        train_parameters['missing_hours'] = m
        evaluate()