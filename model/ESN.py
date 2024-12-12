import pandas as pd
import torch
import torch.sparse
import torch.nn as nn
import numpy as np
import scipy
import os
import torch.multiprocessing as mp
import threading
from copy import deepcopy
from .Metrics import index_agreement_torch, rmse_torch
import io
from npy_append_array import NpyAppendArray

class ESN(nn.Module):

    def __init__(self, **kwargs):
        super(ESN, self).__init__()
        self.load_dict(**kwargs)
        self.training_samples = 0

        self.reservoir_state = torch.zeros(self.reservoir_size, dtype=self.torch_type, device=self.device)

        if self.with_bias:
            self.size = self.reservoir_size + 1
        else:
            self.size = self.reservoir_size

        # ESN matrix generations
        if 'leak_rate' in kwargs.keys():
            self.W = torch.tensor(
                self.generate_reservoir_matrix_liesn(self.reservoir_size, self.connectivity, self.spectral_radius, kwargs['leak_rate']),
            dtype=self.torch_type, device=self.device).to_sparse()
        else:
            self.W = torch.tensor(
                self.generate_reservoir_matrix(self.reservoir_size, self.connectivity, self.spectral_radius),
                dtype=self.torch_type, device=self.device).to_sparse()
        self.W_bias = torch.tensor(
            self.generate_norm_sparce_matrix(self.reservoir_size, 1, self.connectivity, self.bias_scaling),
            dtype=self.torch_type, device=self.device)
        self.W_input = torch.tensor(
            self.generate_norm_sparce_matrix(self.reservoir_size, self.input_dim, self.connectivity,
                                             self.input_scaling), dtype=self.torch_type, device=self.device)
        self.W_out = torch.squeeze(torch.zeros((self.output_dim, self.size), dtype=self.torch_type, device=self.device))

        # Ridge regression parameters
        # X = neuron states
        # Y = desired output
        # W_out = (XX^T - λI)^-1 XY

        self.XX_sum = torch.zeros((self.size, self.size), dtype=self.torch_type, device=self.device)  # Sum (XX^T)
        self.XY_sum = torch.zeros((self.size, self.output_dim), dtype=self.torch_type, device=self.device)  # Sum (XY)

    def load_dict(self, **kwargs):
        self.kwargs = kwargs
        self.input_dim = kwargs['input_dim']
        self.output_dim = kwargs['output_dim']
        self.reservoir_size = kwargs['reservoir_size']
        self.spectral_radius = kwargs['spectral_radius']
        self.bias_scaling = kwargs['bias_scaling']
        self.input_scaling = kwargs['input_scaling']
        self.ridge_parameter = kwargs['ridge_parameter']
        self.connectivity = kwargs['connectivity']
        if 'torch_type' in kwargs.keys():
            self.torch_type = kwargs['torch_type']
        else:
            self.torch_type = torch.float32
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        if 'activation' in kwargs.keys():
            self.activation = kwargs['activation']
        else:
            self.activation = torch.tanh
        if 'loss' in kwargs.keys():
            self.loss = kwargs['loss']
        else:
            self.loss = index_agreement_torch
        if 'seed' in kwargs.keys():
            self.seed = kwargs['seed']
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        if 'sparse' in kwargs.keys():
            self.sparse = kwargs['sparse']
        else:
            self.sparse = True
        if 'with_bias' in kwargs.keys():
            self.with_bias = kwargs['with_bias']
        else:
            self.with_bias = True

    def generate_reservoir_matrix(self, size, density, spectral_radius):
        distribution = scipy.stats.norm().rvs
        matrix = scipy.sparse.random(size, size, density, data_rvs=distribution).toarray()
        largest_eigenvalue = np.max(np.abs(np.linalg.eigvals(matrix)))
        matrix = matrix * spectral_radius / largest_eigenvalue
        return matrix

    def generate_reservoir_matrix_liesn(self, size, density, eq_spectral_radius, leakage):
        distribution = scipy.stats.norm().rvs
        matrix = scipy.sparse.random(size, size, density, data_rvs=distribution).toarray()
        largest_eigenvalue = np.max(np.abs(np.linalg.eigvals(matrix + (1-leakage)*np.eye(size))))
        matrix = (matrix + (1-leakage)*np.eye(size))* eq_spectral_radius / largest_eigenvalue - (1-leakage)*np.eye(size)
        return matrix
    def generate_norm_sparce_matrix(self, rows, columns, density, scaling):
        distribution = scipy.stats.norm().rvs
        matrix = scipy.sparse.random(rows, columns, density, data_rvs=distribution).toarray()
        matrix = matrix * scaling
        return np.squeeze(matrix)

    def set_state_space_size(self, size):
        self.size = size
        self.W_out = torch.squeeze(torch.zeros((self.output_dim, self.size), dtype=self.torch_type, device=self.device))
        self.XX_sum = torch.zeros((self.size, self.size), dtype=self.torch_type, device=self.device)  # Sum (XX^T)
        self.XY_sum = torch.zeros((self.size, self.output_dim), dtype=self.torch_type, device=self.device)  # Sum (XY)

    def save(self, name):
        torch.save((self.W, self.W_bias, self.W_input, self.W_out, self.XX_sum, self.XY_sum, self.training_samples,
                    self.kwargs), name)

    def load(self, name):
        self.W, self.W_bias, self.W_input, self.W_out, self.XX_sum, self.XY_sum, self.training_samples, kwargs = torch.load(
            name, map_location=self.device)
        self.load_dict(**kwargs)

    def forward(self, x):

        self.reservoir_state = self.activation(
            torch.mul(input=self.W_input, other=x.cuda()) + self.W_bias + torch.matmul(self.W, self.reservoir_state))
        output = torch.matmul(input=self.W_out, other=torch.unsqueeze(self.reservoir_state, 1))

        return torch.squeeze(output)

    def load_matrices(self, w, w_in, w_bias, w_out):

        wv = torch.load(w, weights_only=True)
        w_inv = torch.load(w_in, weights_only=True)
        w_bv = torch.load(w_bias, weights_only=True)
        w_ov = torch.load(w_out, weights_only=True)
        if w_inv.shape[0] == w_bv.shape[0] == wv.shape[0] == wv.shape[1]:
            if w_ov.shape[1] == wv.shape[1]:
                self.with_bias = False
            elif w_ov.shape[1] == wv.shape[1] + 1:
                self.with_bias = True
            else:
                return
            self.input_dim = min([min(w_inv.shape),1])
            self.reservoir_size = wv.shape[0]
            self.size = w_ov.shape[1]
            self.W = torch.tensor(wv.clone().detach(), device=self.device, dtype=self.torch_type)
            self.W_bias = torch.tensor(w_bv.clone().detach(), device=self.device, dtype=self.torch_type)
            self.W_input = torch.squeeze(
                torch.tensor(w_inv.clone().detach(), device=self.device, dtype=self.torch_type))
            self.W_out = torch.tensor(w_ov.clone().detach(), device=self.device, dtype=self.torch_type)
        else:
            print("Inconsistent matrix sizes while trying to load tensors from files")
            return
        print('ok')

    def save_weights(self, prefix):
        torch.save(self.W_input, prefix + '_w_in')
        torch.save(self.W, prefix + '_w')
        torch.save(self.W_bias, prefix + '_w_bias')
        torch.save(self.W_out, prefix + '_w_out')

    def load_weights(self, prefix):
        self.W_input = torch.load(prefix + '_w_in', weights_only=True)
        self.W = torch.load(prefix + '_w', weights_only=True)
        self.W_bias = torch.load(prefix + '_w_bias', weights_only=True)
        self.W_out = torch.load(prefix + '_w_out', weights_only=True)

    def predict(self, x, n):
        output = list()
        # give startup time-series as inputs for the reservoir, if there is any nan in the time-series it will use the
        # last value predicted by the reservoir

        esn_initialized = False
        # print('start batch')
        for i in range(x.size(0)):
            if not torch.isnan(x[i]):
                output.append(self.forward(x[i]))
                esn_initialized = True
            elif esn_initialized:
                output.append(self.forward(output[i - 1]))
            else:
                output.append(torch.nan)
            # print(str(i) + ' - ' + str(output[-1].cpu()))
        # print('start forecast')
        for i in range(n):
            output.append(self.forward(output[-1]))
            # print(str(i+x.size(0)) + ' - ' + str(output[-1].cpu()))
        self.reset_internal_state()
        # print('end batch')
        return torch.tensor(output, device=self.device, dtype=self.torch_type)

    def predict_batches(self, batches, future_steps, calc_loss=True):
        predictions = list()
        losses = list()
        ground_truth = list()
        for batch in batches:
            if calc_loss:
                x, y = batch
            else:
                x = batch
            prediction = self.predict(x[0:-future_steps], future_steps)
            predictions.append(prediction)
            if calc_loss:
                losses.append(
                    self.loss(torch.squeeze(prediction[-future_steps:-1]), torch.squeeze(y[-future_steps:-1])))
                ground_truth.append(y)
        return predictions, ground_truth, losses

    def train_batch(self, x, y):
        for inp, out in zip(x, y):
            self.forward(inp)
            self.XX_sum = self.XX_sum + torch.matmul(torch.unsqueeze(self.reservoir_state, 1),
                                                     torch.unsqueeze(self.reservoir_state, 1).t()).t()
            self.XY_sum = self.XY_sum + torch.matmul(torch.unsqueeze(self.reservoir_state, 1),
                                                     torch.unsqueeze(out, 1).t())
            self.training_samples = self.training_samples + 1
            if torch.isnan(self.XX_sum).any() or torch.isnan(self.XY_sum).any():
                raise ("ERROR: NAN DETECTED DURING RIDGE REGRESSION")
        self.reset_internal_state()


    def reset_internal_state(self):
        self.reservoir_state.fill_(0.0)

    def train_epoch(self, dataloader):
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            self.train_batch(batch_x, batch_y)
            print("Trained batch " + str(idx))

    # Workaround for using sparse tensors with multiprocessing
    def train_epoch_multiprocessing(self, dataloader, name):
        self.load(name)
        for idx, batch in enumerate(dataloader):
            batch_x, batch_y = batch
            self.train_batch(batch_x, batch_y)
            print("Trained batch " + str(idx))

    def get_ridge_sums(self):
        return self.XX_sum, self.XY_sum, self.training_samples

    def set_ridge_sums(self, XX_sum, XY_sum, n_samples):
        self.XX_sum = XX_sum
        self.XY_sum = XY_sum
        self.training_samples = n_samples

    def train_finalize(self):
        self.XX_sum = self.XX_sum / self.training_samples
        self.XY_sum = self.XY_sum / self.training_samples
        eye = torch.mul(input=torch.eye(self.reservoir_size, device=self.device), other=self.ridge_parameter)
        eye[0, 0] = 0
        try:
            xx_in = (self.XX_sum + eye).inverse()
        except RuntimeError:
            print('Ridge Regression: matrix has no inverse! Using pseudoinverse instead')
            xx_in = (self.XX_sum + eye).pinverse()
        self.W_out = torch.matmul(input=xx_in, other=self.XY_sum).t()


class LiESN(ESN):
    def __init__(self, **kwargs):
        super(LiESN, self).__init__(**kwargs)
        self.leak_rate = torch.tensor(kwargs['leak_rate'], dtype=self.torch_type, device=self.device)

    def load_dict(self, **kwargs):
        super().load_dict(**kwargs)
        self.leak_rate = torch.tensor(kwargs['leak_rate'], dtype=self.torch_type, device=self.device)

    def forward(self, x, *args, **kwargs):
        self.reservoir_state = torch.mul(
            self.activation(
                torch.mul(input=self.W_input, other=x) + self.W_bias + torch.matmul(self.W, self.reservoir_state)),
            self.leak_rate) \
                               + torch.mul(self.reservoir_state, 1 - self.leak_rate)
        output = torch.matmul(input=self.W_out, other=torch.unsqueeze(self.reservoir_state, 1))

        return torch.squeeze(output)


class LiESNd(LiESN):
    def __init__(self, **kwargs):
        super(LiESNd, self).__init__(**kwargs)
        if 'time_constant' in kwargs.keys():
            self.tc = torch.tensor(kwargs['time_constant'], dtype=self.torch_type, device=self.device)
        else:
            self.tc = torch.tensor(1.0, dtype=self.torch_type, device=self.device)
        if 'output_map' in kwargs.keys():
            self.out_maps = kwargs['output_map']  # Relation between the outputs and inputs
            self.inp_maps = np.empty(self.input_dim)  # Relation between the inputs and outputs
            #self.inp_maps[:] = np.nan
            #for idx, value in enumerate(self.out_maps):
            #    self.inp_maps[value] = idx
        elif 'out_maps' in kwargs.keys():
            self.out_maps = kwargs['out_maps']
            #self.inp_maps = range(self.input_dim)
        else:
            self.out_maps = range(self.input_dim)
            #self.inp_maps = range(self.input_dim)
        self.training_samples = np.zeros(self.output_dim)
        self.XX_sums = list()
        self.XY_sums = list()
        for idx in range(self.output_dim):
            self.XX_sums.append(torch.zeros((self.size, self.size), dtype=self.torch_type, device=self.device))
            self.XY_sums.append(torch.zeros((self.size, 1), dtype=self.torch_type, device=self.device))
        if self.with_bias:
            self.extended_states = torch.cat(
                (torch.tensor([1.0], dtype=self.torch_type, device=self.device), self.reservoir_state))
        else:
            self.extended_states = self.reservoir_state.clone().detach()
        self.ridges = None

    def define_is_size(self, size):
        self.size = size
        self.reset()

    def set_state_space_size(self, size):
        self.size = size
        self.XX_sums = list()
        self.XY_sums = list()
        for idx in range(self.output_dim):
            self.XX_sums.append(torch.zeros((self.size, self.size), dtype=self.torch_type, device=self.device))
            self.XY_sums.append(torch.zeros((self.size, 1), dtype=self.torch_type, device=self.device))
        self.W_out = torch.squeeze(torch.zeros((self.output_dim, self.size), dtype=self.torch_type, device=self.device))

    def define_heterogeneous_ridge(self, own, others):
        self.ridges = own[1] * torch.ones(own[0] + self.with_bias, device=self.device, dtype=self.torch_type)
        for other in others:
            self.ridges = torch.cat(
                (self.ridges, other[1] * torch.ones(other[0], device=self.device, dtype=self.torch_type)))

    def save(self, name):
        torch.save((self.W, self.W_bias, self.W_input, self.W_out, self.XX_sum, self.XY_sum, self.training_samples,
                    self.inp_maps, self.out_maps, self.tc, self.kwargs), name)

    def load(self, name):
        self.W, self.W_bias, self.W_input, self.W_out, self.XX_sum, self.XY_sum, self.training_samples, self.inp_maps, \
        self.out_maps, self.tc, kwargs = torch.load(name, map_location=self.device)
        self.load_dict(**kwargs)

    # forward_w_extended_states(x, dt, extended_states)
    def forward_w_extended_states(self, x, *args, **kwargs):
        dt = args[0] / self.tc
        extended_states = args[1]
        # self.reservoir_state[-2:] = torch.zeros(2, device=self.device, dtype=self.torch_type)
        # self.reservoir_state = torch.mul(
        #         self.activation(torch.matmul(input=self.W_input, other=x) + self.W_bias + torch.matmul(self.W, self.reservoir_state)),dt*self.leak_rate)\
        #         + torch.mul(self.reservoir_state, 1-dt*self.leak_rate)
        while dt > 1.0:
            self.reservoir_state = self.activation(torch.mul(self.W_input, x) + self.W_bias + torch.matmul(self.W, self.reservoir_state)) \
                                   + torch.mul(self.reservoir_state, 1 - self.leak_rate)
            dt = dt - 1.0
        self.reservoir_state = dt*self.activation(torch.mul(self.W_input, x) + self.W_bias + torch.matmul(self.W, self.reservoir_state) \
                                + torch.mul(self.reservoir_state, 1 - dt * self.leak_rate))
        # self.reservoir_state[-2:] = x.detach().clone()
        if self.with_bias:
            self.extended_states = torch.cat(
                (torch.tensor([1.0], dtype=self.torch_type, device=self.device), self.reservoir_state, extended_states))
        else:
            self.extended_states = torch.cat((self.reservoir_state, extended_states))

        output = torch.matmul(input=self.W_out, other=self.extended_states)

        return output

    def forward(self, x, *args, **kwargs):
        dt = args[0] / self.tc
        # self.reservoir_state[-2:] = torch.zeros(2, device=self.device, dtype=self.torch_type)
        # self.reservoir_state = torch.mul(
        #         self.activation(torch.matmul(input=self.W_input, other=x) + self.W_bias + torch.matmul(self.W, self.reservoir_state)),dt*self.leak_rate)\
        #         + torch.mul(self.reservoir_state, 1-dt*self.leak_rate)
        if self.input_dim == 1:
             inputs = torch.mul(self.W_input, x)
        else:
            inputs = torch.matmul(self.W_input, x)
        while dt > 1.0:
            self.reservoir_state = self.activation(inputs + self.W_bias + torch.matmul(self.W, self.reservoir_state))\
                                   + torch.mul(self.reservoir_state, 1 - self.leak_rate)
            dt = dt - 1.0
        self.reservoir_state = torch.mul(
            self.activation(inputs + self.W_bias + torch.matmul(self.W, self.reservoir_state)),
            dt) + torch.mul(self.reservoir_state, 1 - dt * self.leak_rate)
        # self.reservoir_state[-2:] = x.detach().clone()

        if self.with_bias:
            self.extended_states = torch.cat(
                (torch.tensor([1.0], dtype=self.torch_type, device=self.device), self.reservoir_state))
        else:
            self.extended_states = self.reservoir_state.clone().detach()

        output = torch.matmul(input=self.W_out, other=self.extended_states)

        return output

    def generate_tabular_dataset(self, dataloader, b_filename):
        file_idx = 0
        npaa = NpyAppendArray('dummy', delete_if_exists=True)
        for idx, batch in enumerate(dataloader):
            for idy, element in enumerate(batch):
                if idy % 100000 == 0:
                    file_idx += 1
                    filename = b_filename + '_' + str(file_idx)
                    npaa.close()
                    npaa = NpyAppendArray(filename, delete_if_exists=True)
                inp, out = element
                dt = torch.tensor(inp[1], dtype=self.torch_type, device=self.device)
                self.forward(torch.tensor(inp[0], dtype=self.torch_type, device=self.device), dt)
                npaa.append(np.array(self.reservoir_state.cpu(), dtype=np.float32 ))
        npaa.close()

    def train_epoch(self, dataloader):
        for idx, batch in enumerate(dataloader):
            self.train_batch(batch)
            print("Trained batch " + str(idx))
        print('finished_training')

    def train_batch(self, batch, throwaway=None):
        for element in batch:
            inp, out = element
            dt = torch.tensor(inp[1], dtype=self.torch_type, device=self.device)
            self.forward(torch.tensor(inp[0], dtype=self.torch_type, device=self.device), dt)
            out_value, predict, time_out = out
            for idx in range(self.output_dim):
                if predict[idx]:
                    self.add_train_point(idx, torch.tensor([out_value[idx]], dtype=self.torch_type, device=self.device))
                if torch.isnan(self.XX_sums[idx]).any() or torch.isnan(self.XY_sums[idx]).any():
                    raise ("ERROR: NAN DETECTED DURING RIDGE REGRESSION")
        self.reset_internal_state()

    def add_train_point(self, idx, y):
        self.XX_sums[idx] = self.XX_sums[idx] + torch.matmul(torch.unsqueeze(self.extended_states, 1),
                                                             torch.unsqueeze(self.extended_states, 1).t()).t()
        self.XY_sums[idx] = self.XY_sums[idx] + torch.matmul(torch.unsqueeze(self.extended_states, 1),
                                                             torch.unsqueeze(y.t(), 1))
        self.training_samples[idx] = self.training_samples[idx] + 1

    def train_finalize(self):
        Wout = list()
        for idx in range(self.output_dim):
            self.XX_sums[idx] = self.XX_sums[idx] / self.training_samples[idx]
            self.XY_sums[idx] = self.XY_sums[idx] / self.training_samples[idx]
            eye = torch.mul(input=torch.eye(self.size, device=self.device), other=self.ridge_parameter)
            if self.ridges is not None:
                eye[range(self.ridges.shape[0]), range(self.ridges.shape[0])] = self.ridges
            try:
                xx_in = (self.XX_sums[idx] + eye).inverse()
            except RuntimeError:
                print('Ridge Regression: matrix has no inverse! Using pseudoinverse instead')
                xx_in = (self.XX_sums[idx] + eye).pinverse()
            Wout.append(torch.matmul(input=xx_in, other=self.XY_sums[idx]).t())
        self.W_out = torch.cat(tuple(Wout))
        self.W_out = self.W_out.clone().detach()

    def reset(self):
        self.W_out = torch.squeeze(
            torch.zeros((self.output_dim, self.size), dtype=self.torch_type, device=self.device))
        self.reset_internal_state()
        self.training_samples = np.zeros(self.output_dim)
        self.XX_sums = list()
        self.XY_sums = list()
        for idx in range(self.output_dim):
            self.XX_sums.append(torch.zeros((self.size, self.size), dtype=self.torch_type, device=self.device))
            self.XY_sums.append(torch.zeros((self.size, 1), dtype=self.torch_type, device=self.device))

    @torch.no_grad()
    def predict_batches(self, batches, forecast_horizon, warmup, calc_loss=True, return_rmse = False):
        predictions = list()
        losses = list()
        rmses = list()
        inputs = list()
        for idb, batch in enumerate(batches):
            # print('batch: ' + str(idx))
            inp, prediction, comparison_pairs = self.predict(batch, warmup)
            valid = True
            if calc_loss:
                loss = list()
                rmse = list()
                for idx in range(self.output_dim):
                    if len(comparison_pairs[idx]) > 0:
                        loss.append(self.loss(comparison_pairs[idx][:, 0], comparison_pairs[idx][:, 1]).cpu().numpy())
                        rmse.append(rmse_torch(comparison_pairs[idx][:, 0], comparison_pairs[idx][:, 1]).cpu().numpy())
                    else:
                        valid = False
                        loss.append(
                            torch.tensor(np.nan, device=torch.device('cuda'), dtype=torch.float32).cpu().numpy())
                        rmse.append(
                            torch.tensor(np.nan, device=torch.device('cuda'), dtype=torch.float32).cpu().numpy())
                if valid:
                    losses.append(loss)
                    rmses.append(rmse)
            if valid:
                inputs.append(inp)
                predictions.append([prediction, comparison_pairs])
                print("Predicted batch: " + str(idb))
            else:
                print('Droped batch: ' + str(idb) + ' due to lack of data in the forecasting period')
        if return_rmse :
            return inputs, predictions, losses, rmses
        else:
            return inputs, predictions, losses

    @torch.no_grad()
    def predict(self, batch, warmup=7*24, return_internal_states=False):
        output = list()
        inputs = list()
        internal_states = list()
        for idx in range(self.output_dim):
            internal_states.append(list())
        batch.set_warmup(pd.to_timedelta(warmup, unit='hours'))
        LiESN_prediction = None
        prediction_gt_pairs = list()
        self.reset_internal_state()
        for idx in range(self.output_dim):
            prediction_gt_pairs.append(list())

        for element in batch:
            inp, out = element
            dt = torch.tensor(inp[1], dtype=self.torch_type, device=self.device)
            dt_i = dt.clone().detach()
            #inputs.append((inp[0].copy(), dt_i.cpu().numpy(), inp[2].copy()))
            #broi = np.copy(inp[0])  # broi = Best representation of inputs
            inputs.append((inp[0].clone().detach(), dt_i.cpu().clone().detach(), inp[2]))
            broi = inp[0].clone().detach()  # broi = Best representation of inputs
            for idx in range(self.input_dim):
                if torch.isnan(inp[0][idx]):
                    if LiESN_prediction is not None:
                        if not np.isnan(self.out_maps[idx]):
                            broi[idx] = LiESN_prediction[self.out_maps[idx]]
                    else:
                        broi[idx] = 0.0

            while dt_i > self.tc:
                LiESN_prediction = self.forward(broi.detach().clone(),
                                                torch.tensor(1.0, dtype=self.torch_type, device=self.device))
                dt_i = dt_i - self.tc
                for idx in range(len(self.out_maps)):
                    if not np.isnan(self.out_maps[idx]):
                        broi[idx] = LiESN_prediction[self.out_maps[idx]]

            LiESN_prediction = self.forward(broi.detach().clone(), dt_i)

            if any(torch.isnan(LiESN_prediction)):
                print('STOP')

            output.append([LiESN_prediction.detach().clone(), dt.detach().clone(), inp[2]])
            out_value, predict, time = out
            # print(str(predict) + ' - ' + str(inp[2]))
            for idx in range(self.output_dim):

                if predict[idx]:
                    prediction_gt_pairs[idx].append([LiESN_prediction[idx].detach().clone(),
                                                     out_value[idx].detach().clone()])

                if return_internal_states:
                    if any(predict):
                        internal_states[idx].append(self.reservoir_state.clone().detach())

        list_pred = list()
        for pred in prediction_gt_pairs:
            list_pred.append(torch.tensor(pred, dtype=self.torch_type, device=self.device))

        prediction_gt_pairs = tuple(list_pred)
        if return_internal_states:
            return inputs, output, prediction_gt_pairs, internal_states
        else:
            return inputs, output, prediction_gt_pairs

    def warm(self, batch, return_internal_states=True):
        output = list()
        inputs = list()
        internal_states = list()
        prediction_gt_pairs = list()
        for idx in range(self.output_dim):
            internal_states.append(list())
        LiESN_prediction = None
        self.reset_internal_state()

        for element in batch:
            inp, out = element
            dt = torch.tensor(inp[1], dtype=self.torch_type, device=self.device)
            dt_i = dt.clone().detach()
            inputs.append((inp[0].copy(), dt_i.cpu().numpy(), inp[2].copy()))
            broi = np.copy(inp[0])  # broi = Best representation of inputs
            for idx in range(self.input_dim):
                if np.isnan(inp[0][idx]):
                    if LiESN_prediction is not None:
                        if not np.isnan(self.out_maps[idx]):
                            broi[idx] = LiESN_prediction[self.out_maps[idx]]
                    else:
                        broi[idx] = 0.0

            while dt_i > self.tc:
                LiESN_prediction = self.forward(torch.tensor(broi, dtype=self.torch_type, device=self.device),
                                                torch.tensor(1.0, dtype=self.torch_type, device=self.device))
                dt_i = dt_i - self.tc
                for idx in range(self.input_dim):
                    if not np.isnan(self.out_maps[idx]):
                        broi[idx] = LiESN_prediction[self.out_maps[idx]]

            LiESN_prediction = self.forward(torch.tensor(broi, dtype=self.torch_type, device=self.device), dt_i)
            output.append([LiESN_prediction.detach().clone(), dt.detach().clone(), inp[2]])
            out_value, predict, time = out
            # print(str(predict) + ' - ' + str(inp[2]))
            for idx in range(self.output_dim):
                if predict[idx]:
                    prediction_gt_pairs[idx].append([LiESN_prediction[idx].detach().clone(),
                                                     torch.tensor(out_value[idx], dtype=self.torch_type,
                                                                  device=self.device)])
                if return_internal_states:
                    if any(predict):
                        internal_states[idx].append(self.reservoir_state.clone().detach())

        list_pred = list()
        for pred in prediction_gt_pairs:
            list_pred.append(torch.tensor(pred, dtype=self.torch_type, device=self.device))

        prediction_gt_pairs = tuple(list_pred)

        return inputs, output, prediction_gt_pairs, internal_states


class DeepLiESNd:
    def __init__(self, **kwargs):
        self.layers = dict()
        self.layer_sequence = list()
        self.internal_size = 0
        self.input_map = kwargs['input_map']
        for idl, layer in kwargs["layers"].items():
            self.layers[idl] = dict()
            self.layer_sequence.append(idl)
            self.layers[idl]["reservoirs"] = dict()
            self.layers[idl]["dicts"] = dict()
            for idr, reservoir in layer.items():
                reservoir['seed'] = kwargs['seed']
                print("Using seed: " + str(reservoir['seed']))
                self.layers[idl]["reservoirs"][idr] = LiESNd(**reservoir)
                self.layers[idl]["dicts"][idr] = reservoir
                self.internal_size += reservoir['reservoir_size']
                self.input_map[idr] = range(self.layers[idl]["dicts"][idr]['input_dim'])

        self.ridge = kwargs['ridge']
        self.device = kwargs["device"]
        self.dtype = kwargs["torch_type"]
        self.with_bias = kwargs["use_bias"]
        self.input_dim = kwargs["input_dim"]
        self.tc = kwargs["time_constant"]
        self.output_dim = kwargs["output_dim"]
        self.out_maps = kwargs["out_maps"]
        self.loss = kwargs["loss"]
        self.internal_states = None
        self.extended_states = None
        self.ridges = None

        if self.with_bias:
            self.internal_size += 1

        self.training_samples = np.zeros(self.output_dim)
        self.XX_sums = list()
        self.XY_sums = list()
        for idx in range(self.output_dim):
            self.XX_sums.append(torch.zeros((self.internal_size, self.internal_size), dtype=self.dtype, device=self.device))
            self.XY_sums.append(torch.zeros((self.internal_size, 1), dtype=self.dtype, device=self.device))

        self.W_out = torch.squeeze(
            torch.zeros((self.output_dim, self.internal_size), dtype=self.dtype, device=self.device))


    def save(self, name):

        os.makedirs(name, exist_ok=True)

        torch.save((self.ridge, self.device, self.dtype, self.with_bias, self.input_dim, self.tc, self.output_dim,
                    self.out_maps, self.loss, self.input_map, self.input_map, self.W_out, self.layers), name + '/params')

        for lkey, layer in self.layers.items():
            reservoirs = layer["reservoirs"]
            for rkey, reservoir in reservoirs.items():
                reservoir.save(name + '/' + rkey)

    def load(self, name):

        self.ridge, self.device, self.dtype, self.with_bias, self.input_dim, self.tc, self.output_dim, self.out_maps, self.loss, self.input_map, self.input_map, self.W_out, self.layers = torch.load(name + '/params', map_location=self.device)
        for lkey, layer in self.layers.items():
            reservoirs = layer["reservoirs"]
            for rkey, reservoir in reservoirs.items():
                reservoir.load(name + '/' + rkey)

    def save_weights(self, path):
        for idl, layer in self.layers.items():
            for idr, reservoir in layer['reservoirs'].items():
                self.layers[idl]["reservoirs"][idr].save_weights(path + '_' +idr + '_')
        torch.save(self.W_out, path + '_trained_weights')

    def load_weights(self, path):
        for idl, layer in self.layers.items():
            for idr, reservoir in layer['reservoirs'].items():
                self.layers[idl]["reservoirs"][idr].load_weights(path + '_' +idr + '_')
        try:
            self.W_out = torch.load(path + '_trained_weights', weights_only=True)
        except:
            print("Failed to load trained weights")

    def forward(self, x, *args, **kwargs):
        dt = args[0]
        this_input = x
        self.internal_states = list()
        for idl, layer in self.layers.items():
            next_inputs = list()
            for idr, reservoir in layer['reservoirs'].items():
                self.layers[idl]["reservoirs"][idr].forward(this_input[self.input_map[idr]], dt)
                next_inputs.append( self.layers[idl]["reservoirs"][idr].reservoir_state)
                self.internal_states.append(self.layers[idl]["reservoirs"][idr].reservoir_state)
            this_input = torch.cat(tuple(next_inputs), 0)

        self.internal_states = torch.cat(tuple(self.internal_states), 0)

        if self.with_bias:
            self.extended_states = torch.cat(
                (torch.tensor([1.0], dtype=self.dtype, device=self.device), self.internal_states))
        else:
            self.extended_states = self.internal_states.clone().detach()

        return torch.matmul(input=self.W_out, other=self.extended_states)

    def add_train_point(self, idx, y):
        self.XX_sums[idx] = self.XX_sums[idx] + torch.matmul(torch.unsqueeze(self.extended_states, 1),
                                                             torch.unsqueeze(self.extended_states, 1).t()).t()
        self.XY_sums[idx] = self.XY_sums[idx] + torch.matmul(torch.unsqueeze(self.extended_states, 1),
                                                             torch.unsqueeze(y.t(), 1))
        self.training_samples[idx] = self.training_samples[idx] + 1

    def train_finalize(self):
        Wout = list()
        for idx in range(self.output_dim):
            self.XX_sums[idx] = self.XX_sums[idx] / self.training_samples[idx]
            self.XY_sums[idx] = self.XY_sums[idx] / self.training_samples[idx]
            eye = torch.mul(input=torch.eye(self.internal_size, device=self.device), other=self.ridge)
            if self.ridges is not None:
                eye[range(self.ridges.shape[0]), range(self.ridges.shape[0])] = self.ridges
            try:
                xx_in = (self.XX_sums[idx] + eye).inverse()
            except RuntimeError:
                print('Ridge Regression: matrix has no inverse! Using pseudoinverse instead')
                xx_in = (self.XX_sums[idx] + eye).pinverse()
            Wout.append(torch.matmul(input=xx_in, other=self.XY_sums[idx]).t())
        self.W_out = torch.cat(tuple(Wout))
        self.W_out = self.W_out.clone().detach()

    def train_epoch(self, dataloader):
        for idx, batch in enumerate(dataloader):
            self.train_batch(batch)
            print("Trained batch " + str(idx))


    def train_batch(self, batch, throwaway=None):
        for element in batch:
            inp, out = element
            dt = torch.tensor(inp[1], dtype=self.dtype, device=self.device)
            self.forward(torch.tensor(inp[0], dtype=self.dtype, device=self.device), dt)
            out_value, predict, time_out = out
            for idx in range(self.output_dim):
                if predict[idx]:
                    self.add_train_point(idx, torch.tensor([out_value[idx]], dtype=self.dtype, device=self.device))

        self.reset_internal_state()

    def reset_internal_state(self):
        for idl, layer in self.layers.items():
            for idr, reservoir in layer["reservoirs"] .items():
                reservoir.reset_internal_state()

    def reset(self):
        self.W_out = torch.squeeze(
            torch.zeros((self.output_dim, self.internal_size), dtype=self.dtype, device=self.device))
        self.reset_internal_state()
        self.training_samples = np.zeros(self.output_dim)
        self.XX_sums = list()
        self.XY_sums = list()
        for idx in range(self.output_dim):
            self.XX_sums.append(torch.zeros((self.internal_size, self.internal_size), dtype=self.dtype, device=self.device))
            self.XY_sums.append(torch.zeros((self.internal_size, 1), dtype=self.dtype, device=self.device))

    @torch.no_grad()
    def predict_batches(self, batches, forecast_horizon, warmup, calc_loss=True, return_rmse=False):
        predictions = list()
        losses = list()
        rmses = list()
        inputs = list()
        for idb, batch in enumerate(batches):
            # print('batch: ' + str(idx))
            inp, prediction, comparison_pairs = self.predict(batch, warmup)
            valid = True
            if calc_loss:
                loss = list()
                rmse = list()
                for idx in range(self.output_dim):
                    if len(comparison_pairs[idx]) > 0:
                        loss.append(self.loss(comparison_pairs[idx][:, 0], comparison_pairs[idx][:, 1]).cpu().numpy())
                    else:
                        valid = False
                        loss.append(
                            torch.tensor(np.nan, device=torch.device('cuda'), dtype=torch.float32).cpu().numpy())
                        rmse.append(
                            torch.tensor(np.nan, device=torch.device('cuda'), dtype=torch.float32).cpu().numpy())
                if valid:
                    losses.append(loss)
                    rmses.append(rmse)
            if valid:
                inputs.append(inp)
                predictions.append([prediction, comparison_pairs])
                print("Predicted batch: " + str(idb))
            else:
                print('Droped batch: ' + str(idb) + ' due to lack of data in the forecasting period')
        if return_rmse:
            return inputs, predictions, losses, rmses
        else:
            return inputs, predictions, losses

    @torch.no_grad()
    def predict(self, batch, warmup):
        output = list()
        inputs = list()
        batch.set_warmup(pd.to_timedelta(warmup, unit='hours'))
        LiESN_prediction = None
        prediction_gt_pairs = list()
        self.reset_internal_state()
        for idx in range(self.output_dim):
            prediction_gt_pairs.append(list())

        for be, element in enumerate(batch):
            inp, out = element
            dt = torch.tensor(inp[1], dtype=self.dtype, device=self.device)
            dt_i = dt.clone().detach()
            inputs.append((inp[0].clone().detach(), dt_i.cpu().numpy(), inp[2]))
            broi = inp[0].clone().detach()  # broi = Best representation of inputs
            for idx in range(self.input_dim):
                if torch.isnan(inp[0][idx]):
                    if LiESN_prediction is not None:
                        if not np.isnan(self.out_maps[idx]):
                            broi[idx] = LiESN_prediction[self.out_maps[idx]]
                        else:
                            broi[idx] = 0.0
                    else:
                        broi[idx] = 0.0

            while dt_i > self.tc:
                LiESN_prediction = self.forward(torch.tensor(broi, dtype=self.dtype, device=self.device),
                                                torch.tensor(1.0, dtype=self.dtype, device=self.device))
                dt_i = dt_i - self.tc
                for idx in range(self.input_dim):
                    if idx < len(LiESN_prediction):
                        if not np.isnan(self.out_maps[idx]):
                            broi[idx] = LiESN_prediction[self.out_maps[idx]]
                        else:
                            broi[idx] = 0.0

            for idx in range(self.input_dim):
                if torch.isnan(broi):
                    print('Vish')

            LiESN_prediction = self.forward(broi.clone().detach(), dt_i)
            output.append([LiESN_prediction.detach().clone(), dt.detach().clone(), inp[2]])
            out_value, predict, time = out
            # print(str(predict) + ' - ' + str(inp[2]))
            for idx in range(self.output_dim):
                if predict[idx]:
                    prediction_gt_pairs[idx].append([LiESN_prediction[idx].detach().clone(), out_value[idx].clone().detach()])

        list_pred = list()
        for pred in prediction_gt_pairs:
            list_pred.append(torch.tensor(pred, dtype=self.dtype, device=self.device))

        prediction_gt_pairs = tuple(list_pred)

        return inputs, output, prediction_gt_pairs
