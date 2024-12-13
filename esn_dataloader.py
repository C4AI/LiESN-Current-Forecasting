from scipy import signal, stats
import torch
import pandas as pd
import numpy as np
import itertools
import math
from scipy.stats import norm


class Timeseries:
    def __init__(self, **kwargs):
        self.type = kwargs['type']
        if 'description' in kwargs.keys():
            self.description = kwargs['description']
        else:
            self.description = ''
        self.path = kwargs['path']
        self.transformations = list()
        if 'transformations' in kwargs.keys():
            self.transformations = kwargs['transformations']
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device('cpu')

        if 'columns' in kwargs.keys():
            self.columns = kwargs['columns']
        else:
            self.columns = [kwargs['type']]

        if 'is_forecast' in kwargs.keys():
            self.is_forecast = kwargs['is_forecast']
        else:
            self.is_forecast = False

        self.is_predicted = kwargs['is_predicted']
        self.df = None
        self.original_time_series = None
        self.processed_time_series = None

        self.mean = None
        self.std = None

        self.dt = None
        self.mode_dt = None

        self.outputs = (None, None, None)

        self.inputs = list()
        self.outputs = list()
        if 'numpy_ts' in kwargs.keys():
            self.load_data_np(kwargs['numpy_ts'])
        else:
            self.load_data()



    def load_data_np(self, timestamps):
        data = np. load(self.path)
        timestamps = np.load(timestamps)
        steps, res_siz = len(timestamps), int(len(data) / len(timestamps))
        if steps != len(data):
            data = np.reshape(data, (steps,res_siz))
        df = pd.DataFrame(data)
        df['datetime'] = timestamps
        df['available_since'] = timestamps
        df.drop_duplicates(subset='datetime', inplace=True)
        df.sort_values(by='datetime', inplace=True)

        #self.df = df
        for transformation in self.transformations:
            self.apply_transformation(transformation)
        df['timedelta'] = df['datetime'].shift(-1) - df['datetime']
        self.mode_dt = df['timedelta'].mode()
        self.inputs = (df.drop(columns=['datetime', 'available_since', 'timedelta']).to_numpy(),
                                      df['datetime'].values, df['available_since'].values)
        if self.is_predicted:
            self.outputs = (torch.tensor(self.df[self.type].shift(-1).values[0:-1], device=self.device, dtype=torch.float32),
                                      self.df['datetime'].shift(-1).values[0:-1], self.df['available_since'].shift(-1).values[0:-1])

    def load_data(self):
        if '.csv' in self.path:
            df = pd.read_csv(self.path, encoding='latin-1')
        elif '.parquet' in self.path:
            df = pd.read_parquet(self.path)
        else:
            raise Exception('File format not supported')
        df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', yearfirst=True)
        if 'available_since' in df:
            df['available_since'] = pd.to_datetime(df['available_since'], format='mixed', yearfirst=True)
        else:
            df['available_since'] = df['datetime']
        columns_to_filter = self.columns.copy()
        columns_to_filter.append('datetime')
        columns_to_filter.append('available_since')
        df = df[columns_to_filter]
        df.drop_duplicates(subset='datetime', inplace=True)
        df.sort_values(by='datetime', inplace=True)
        df.dropna(inplace=True)

        self.df = df[columns_to_filter]
        self.original_time_series = (df[self.columns].values, df['datetime'], df['available_since'])
        for transformation in self.transformations:
            self.apply_transformation(transformation)

        df['timedelta'] = df['datetime'].shift(-1) - df['datetime']
        self.mode_dt = df['timedelta'].mode()

        self.df.dropna(inplace=True)

        self.inputs = (torch.tensor(self.df[self.columns].values[0:-1], device=self.device, dtype=torch.float32),
                                      self.df['datetime'].values[0:-1], self.df['available_since'].values[0:-1])
        if self.is_predicted:
            self.outputs = (torch.tensor(self.df[self.columns].shift(-1).values[0:-1], device=self.device, dtype=torch.float32),
                                          self.df['datetime'].shift(-1).values[0:-1], self.df['available_since'].shift(-1).values[0:-1])

    # replicates the last value before the start time adding the time encoding at the timestamps provided
    def replicate_at_timestamps(self, start, timestamps):
        valid = False
        for transformation in self.transformations:
            if transformation[0] == 'time_encode':
                valid = True
                break
        if not valid:
            raise Exception("Time encoding not available for time series: " + self.description)

        tw = self.get_time_window(self.inputs[1][0], start)
        last_values = tw[0][-1]
        last_time = tw[1][-1]
        te_base = transformation[1]
        te_dim = transformation[2]
        out = last_values.repeat(len(timestamps),1)
        for idx, tstamp in enumerate(timestamps):
            dt = (tstamp - last_time)/ np.timedelta64(1, 's')
            for dim in range(te_dim):
                denon = np.power(te_base, 2 * math.floor(dim / 2) / te_dim)
                if dim % 2 == 0:
                    out[idx, len(last_values)-te_dim+dim] = torch.tensor(np.sin(dt/denon))
                else:
                    out[idx, len(last_values)-te_dim+dim] = torch.tensor(np.cos(dt/denon))

        return out



    def get_dimension(self):
        return self.inputs[0].shape[1]

    def apply_transformation(self, transformation):
        if transformation[0] == 'low_pass_filter':
            self.lowpass_filter(transformation[1])
        elif transformation[0] == 'z-score':
            if len(transformation) > 1:
                self.z_score(transformation[1], transformation[2])
        elif transformation[0] == 'time_encode':
            self.te(transformation[1],transformation[2])
        elif transformation == 'z-score':
            self.z_score()

    def z_score_reverse(self, data):
        return data*self.std + self.mean

    def z_score(self, mean=None, std=None):
        if mean is not None:
            self.mean = mean
        else:
            self.mean = self.df[self.type].mean()
            print('mean:' + str(self.mean))
        if std is not None:
            self.std = std
        else:
            self.std = self.df[self.type].std()
            print('std:' + str(self.std))

        self.df[self.type] = (self.df[self.type] - self.mean)/self.std

    def te(self, te_base, te_dim):
        dt = (self.df['datetime'].shift(-1) - self.df['datetime']).to_numpy()/np.timedelta64(1, 's')
        for dim in range(te_dim):
            denon = np.power(te_base, 2*math.floor(dim/2) / te_dim)
            if dim % 2 == 0:
                self.df['te_' + str(dim)] = np.sin(dt/denon)
                self.columns.append('te_' + str(dim))
            else:
                self.df['te_' + str(dim)] = np.cos(dt/denon)
                self.columns.append('te_' + str(dim))


    def lowpass_filter(self, cutoff_hours):
        df = self.df
        df_orig = pd.DataFrame(data = {'times' : df['datetime']})
        df['timedelta'] = df['datetime'].shift(-1) - df['datetime']
        resample_freq = df['timedelta'].mode()  # Resample by the most common period between data
        times = pd.DataFrame(
            {'Time': pd.date_range(start=df['datetime'].iloc[0], end=df['datetime'].iloc[-1],
                                   freq=resample_freq.iloc[0])})  # Regular time-series

        # Cast to datetime64 in ns both timestamps
        times['Time'] = pd.to_datetime(times['Time'], utc=True, unit='ns').astype('datetime64[ns, UTC]')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ns').astype('datetime64[ns, UTC]')
        df = pd.merge_asof(left=times, right=df, right_on='datetime', left_on='Time',
                           tolerance=resample_freq.iloc[0], direction='nearest')  # Merging

        df['is_na'] = pd.isna(df[self.type])
        df.ffill(inplace=True)
        cutoff = cutoff_hours / (60 * 60)
        fs = 1 / resample_freq.iloc[0].seconds
        nyquist = fs / 2  # half the sampling frequency
        cutoff = 0.5 * cutoff / fs  # fraction of nyquist frequency
        print('cutoff= ', 1 / (cutoff * nyquist * 3600), ' hours')
        b, a = signal.butter(5, cutoff, btype='lowpass', output='ba')  # low pass filter
        dUfilt = signal.filtfilt(b, a, df[self.type])
        df[self.type] = pd.DataFrame(dUfilt)
        df.loc[df['is_na'] == True, self.type] = np.NaN
        df.dropna(inplace=True)
        df.drop_duplicates(subset='datetime', inplace=True)
        df.drop(columns=["Time"], inplace=True)

        df_orig['times'] = pd.to_datetime(df_orig['times'], utc=True, unit='ns').astype('datetime64[ns, UTC]')
        df_orig = pd.merge_asof(left=df_orig, right=df, right_on='datetime', left_on='times',
                           tolerance=resample_freq.iloc[0], direction='nearest')  # Merging

        self.df[self.type] = df_orig[self.type]

    def get_time_window(self, start, end, drop_first=0, drop_last=0, by_available_since=False, mdtp=None):
        if by_available_since:
            idx = np.logical_and(start <= self.inputs[2], self.inputs[2] <= end)
        else:
            idx = np.logical_and(start <= self.inputs[1], self.inputs[1] <= end)
        if len(idx) == 0:
            print('Empty time window!!')
            return torch.zeros(1, device=self.device, dtype=torch.float32), start, start
        else:
            if drop_last > 0:
                tims = (self.inputs[0][idx][drop_first:-drop_last], self.inputs[1][idx][drop_first:-drop_last], self.inputs[2][idx][drop_first:-drop_last])
            else:
                tims = (self.inputs[0][idx][drop_first:], self.inputs[1][idx][drop_first:], self.inputs[2][idx][drop_first:])

            if mdtp is not None:
                mdt, std, miss_per, for_window = mdtp
                for_window = pd.to_timedelta(for_window, unit='hours')
                ors = len(tims[0])
                while len(tims[0]) > miss_per*ors:
                    r = norm.rvs(mdt,std,size=1)
                    minutes = (end - start - for_window).total_seconds() / 60 - r
                    st = start + pd.Timedelta(int(np.random.randint(low=0, high=minutes, size=1)), unit='minutes')
                    ed = st + pd.Timedelta(int(r), unit='minutes')
                    idx = np.logical_and(tims[1] >= st, tims[1] <= ed)
                    tims[0][idx] = torch.tensor(np.nan, device=self.device, dtype=torch.float32)
                    #idx = np.logical_or(ed <= tims[1], tims[1] <= st)
                    #tims = tims[0][idx], tims[1][idx], tims[2][idx]

            return tims

    def get_time_window_w_missing_before_forecast(self, start, end, for_window, h, drop_first=0, drop_last=0, by_available_since=False):
        if by_available_since:
            idx = np.logical_and(start <= self.inputs[2], self.inputs[2] <= end)
        else:
            idx = np.logical_and(start <= self.inputs[1], self.inputs[1] <= end)
        if len(idx) == 0:
            print('Empty time window!!')
            return torch.zeros(1, device=self.device, dtype=torch.float32), start, start
        else:
            if drop_last > 0:
                tims = (self.inputs[0][idx][drop_first:-drop_last], self.inputs[1][idx][drop_first:-drop_last], self.inputs[2][idx][drop_first:-drop_last])
            else:
                tims = (self.inputs[0][idx][drop_first:], self.inputs[1][idx][drop_first:], self.inputs[2][idx][drop_first:])
            if h > 0:

                ed = end - pd.to_timedelta(for_window, unit='hours')
                st = ed - pd.Timedelta(h, unit='hours')

                idx = np.logical_and(tims[1] >= st, tims[1] <= ed)
                tims[0][idx] = torch.tensor(np.nan, device=self.device, dtype=torch.float32)
                #idx = np.logical_or(ed <= tims[1], tims[1] <= st)
                #tims = tims[0][idx], tims[1][idx], tims[2][idx]

            return tims

    def get_boundaries(self):
        return self.inputs[1].min(), self.inputs[1].max()


    def get_typical_dt(self):
        return self.mode_dt

    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.inputs[0])

    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        return (self.inputs[0][idx], self.inputs[1][idx], self.inputs[2][idx]), (self.outputs[0][idx], self.outputs[1][idx])


class BatchParser:
    def __init__(self, batch, desc, is_input, is_predicted, inp_map, dtype, washout=pd.to_timedelta(0, unit='hours'), warmup=pd.to_timedelta(0, unit='days'), device=torch.device('cpu'), **kwargs):
        self.batch = batch
        self.device = device
        self.dtype = dtype
        self.description = desc
        self.predict = np.array(is_predicted)
        self.inp_map = inp_map
        self.is_input = np.array(is_input)
        self.n_series = len(self.batch)
        self.current_time = None
        self.mask_unavailable = False
        self.washout = washout
        self.warmup = warmup
        self.inference_time = pd.to_datetime(3162240000, unit='s')
        self.inputs = list([torch.zeros(self.is_input.sum(), device=self.device, dtype=self.dtype), np.zeros(1), pd.to_datetime(3162240000, unit='s')])
        self.outputs = list([torch.zeros(self.predict.sum(), device=self.device, dtype=self.dtype), np.zeros(self.predict.sum()), pd.to_datetime(3162240000, unit='s')])
        self.last_time = list()
        self.modes = list()
        self.bypass_zoh = False
        for idx in range(self.n_series):
            dt = self.batch[idx][1][1:] - self.batch[idx][1][:-1]
            mode = stats.mode(dt)
            self.modes.append(pd.to_timedelta(mode.mode))
            self.last_time.append(pd.to_datetime(0, unit='s'))
        if 'time_encode' in kwargs.items():
            self.te = True
            self.te_length = kwargs['length']
            self.te_base = kwargs['base']
            self.inputs = list([np.zeros(self.n_series), np.zeros(1), pd.to_datetime(3162240000, unit='s'), torch.zeros(self.te_length)])
        else:
            self.te = False

    def set_inference_time(self, inference_time):
        self.inference_time = inference_time

    def get_inference_time(self):
        return self.inference_time

    def set_warmup(self, warmup):
        self.warmup = warmup
        current_time = pd.to_datetime(3162240000, unit='s')
        for idx, time_series in enumerate(self.batch):
            if time_series[-2][0] <= current_time:
                current_time = time_series[-2][0]
        self.inference_time = current_time + warmup

    def get_descriptions_ordered(self):
        return self.description

    def mask_unavailable(self, flag):
        self.mask_unavailable = flag

    def __iter__(self):
        self.idx_iter = torch.zeros(self.n_series, device=self.device, dtype=self.dtype)
        self.lengths = [len(x[0]) for x in self.batch]
        self.available_next = torch.ones(self.n_series, device=self.device, dtype=self.dtype)
        self.contin = True
        self.current_time = pd.to_datetime(3162240000, unit='s')
        self.current_available_since = None
        self.next_time = pd.to_datetime(3162240000, unit='s')
        self.start_time = None
        self.current_idx = None
        self.next_idx = None
        return self

    def __next__(self):
        if self.contin:
            self.current_time = pd.to_datetime(3162240000, unit='s')
            self.current_available_since = None
            for idx, time_series in enumerate(self.batch):
                if self.available_next[idx]:
                    if time_series[-2][int(self.idx_iter[idx])] <= self.current_time:
                        self.current_time = time_series[-2][int(self.idx_iter[idx])]
                        self.current_available_since = time_series[-1][int(self.idx_iter[idx])]
                        self.current_idx = idx

            if self.start_time is None:
                self.start_time = self.current_time

            if self.is_input[self.current_idx]:
                if self.current_available_since < self.inference_time:
                    self.inputs[0][self.inp_map[self.current_idx]] = self.batch[self.current_idx][0][int(self.idx_iter[self.current_idx])].clone().detach()
                    self.inputs[2] = self.current_time
                else:
                    self.inputs[0][self.inp_map[self.current_idx]] = torch.nan
                    self.inputs[2] = self.current_time

            self.last_time[self.current_idx] = self.inputs[2]

            # If there is a gap in a predicted time series, stop using the zero order hold, so the model can fill these
            # gaps with its own predictions
            if self.bypass_zoh:
                for idx, _ in enumerate(self.batch):
                    if self.predict[idx] and not np.isnan(self.inp_map[idx]):
                        if self.current_time - self.last_time[idx] > 2*self.modes[idx]:
                            self.inputs[0][self.inp_map[idx]] = torch.tensor(np.nan, device=self.device, dtype=self.dtype)

            self.outputs[1] = np.zeros(len(self.predict))

            self.idx_iter[self.current_idx] = self.idx_iter[self.current_idx] + 1
            if self.idx_iter[self.current_idx] >= self.lengths[self.current_idx]:
                self.available_next[self.current_idx] = False

            self.next_idx = None
            self.next_time = pd.to_datetime(3162240000, unit='s')
            for idx, time_series in enumerate(self.batch):
                if self.available_next[idx]:
                    if time_series[-2][int(self.idx_iter[idx])] <= self.next_time:
                        self.next_time = time_series[-2][int(self.idx_iter[idx])]
                        self.next_idx = idx

            if self.next_time == np.datetime64('2021-12-31T12:50:00.000000000'):
                print('pause')

            if self.current_time - self.start_time >= self.washout and self.current_time - self.start_time >= self.warmup:
                if self.next_idx is not None:
                    predict = self.predict[self.next_idx]
                    if predict:
                        self.outputs[0][self.next_idx] = self.batch[self.next_idx][0][int(self.idx_iter[self.next_idx])].clone().detach()
                        self.outputs[1][self.next_idx] = True
                        self.outputs[2] = self.next_time

            dt = self.next_time - self.current_time
            self.inputs[1] = dt / np.timedelta64(1, 's')
                # print('Time_now: ' + str(self.current_time))
                # print('TS = ' + self.description[self.current_idx])
                # print('DT = ' + str(self.inputs[1]) + 'S')
                # print('Predict? ' + str(self.outputs[1]))
            if self.te:
                self.inputs[3] = torch.tensor([np.sin(dt/self.te_base**(idx/self.te_length)) for idx in range(self.te_length)], device=self.device, dtype=self.dtype)

            if self.available_next.sum() == 0:
                self.contin = False
                dt = self.next_time - self.next_time
                self.inputs[1] = dt / np.timedelta64(1, 's')
            return self.inputs, self.outputs
        else:
            raise StopIteration


class BaseDataLoader:
    def __init__(self, ts, desc, is_predicted, inp_map, device=torch.device('cpu'), dtype=torch.float32, **kwargs):
        self.ts = ts
        self.description = desc
        self.is_predicted = is_predicted
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs
        self.indexes=None
        if 'is_input' in kwargs.keys():
            self.is_input = np.array(kwargs['is_input'])
        else:
            self.is_input = None
        self.inp_map = inp_map


    def set_indexes(self, idx):
        self.indexes = idx

    def __iter__(self):
        self.idx = -1
        return self

    def __next__(self):
        self.idx = self.idx + 1
        if self.idx < len(self.ts):
            if self.indexes is None:
                if self.is_input is not None:
                    return BatchParser(self.ts[self.idx], self.description, self.is_input, self.is_predicted, self.inp_map, device=self.device,
                                       dtype=self.dtype)

            else:
                if self.is_input is not None:
                    return BatchParser(self.ts[self.idx], self.description, self.is_input, self.is_predicted, self.inp_map,
                                       device=self.device, dtype=self.dtype)
        else:
            raise StopIteration


class DataSplitter:
    def __init__(self, **kwargs):
        self.timeseries = list()
        self.types = list()
        self.descriptions = list()
        self.is_predicted = list()
        self.allow_missing = list()
        self.missing_threshold = list()
        self.batches = None
        self.inp_dim = 0
        self.out_dim = 0
        self.out_map = list()
        self.descriptions_map = dict()
        self.is_input = list()
        self.inp_map = list()
        for idx, timeseries in enumerate(kwargs['timeseries']):
            self.timeseries.append(Timeseries(**timeseries))
            self.types.append(timeseries['type'])
            self.is_predicted.append(timeseries['is_predicted'])
            self.descriptions.append(self.timeseries[-1].description)
            if self.timeseries[-1].description in self.descriptions_map.keys():
                self.descriptions_map[self.timeseries[-1].description] = np.array([self.descriptions_map[self.timeseries[-1].description], idx ])
            else:
                self.descriptions_map[self.timeseries[-1].description] = np.array(idx)
            if 'allow_missing' in timeseries:
                self.allow_missing.append(timeseries['allow_missing'])
            else:
                self.allow_missing.append(True)
            if 'missing_threshold' in timeseries:
                self.missing_threshold.append(pd.Timedelta(timeseries['missing_threshold'], unit='minutes'))
            else:
                self.missing_threshold.append(None)
            if timeseries['is_predicted']:
                self.out_dim = self.out_dim + 1
            if 'is_input' in timeseries.keys():
                self.is_input.append(timeseries['is_input'])
            else:
                self.is_input.append(True)
            if self.is_input[-1]:
                self.inp_dim = self.inp_dim + 1
                if self.is_predicted[-1]:
                    self.out_map.append(np.array(self.is_predicted).sum()-1)
                else:
                    self.out_map.append(np.nan)
                self.inp_map.append(np.array(self.is_input).sum()-1)
            else:
                self.inp_map.append(np.nan)

        if 'time_encoding' in kwargs.keys():
            self.batch_dict = kwargs['time_encoding']
        else:
            self.batch_dict = dict()

        self.batch_dict['is_input'] = self.is_input

        if 'batch_duration' in kwargs.keys() and 'stride' in kwargs.keys():
            self.create_batches(kwargs['batch_duration'], kwargs['stride'])

        self.device = kwargs['device']
        self.dtype = kwargs['dtype']

        if 'train_batches' in kwargs.keys() and 'val_batches' in kwargs.keys():
            self.train_batches = kwargs['train_batches']
            self.val_batches = kwargs['val_batches']
        else:
            self.train_batches = None
            self.val_batches = None

        if self.train_batches is None:
            self.train_batches = 10
        if self.val_batches is None:
            self.val_batches = 10


    def create_batches(self, batch_duration, stride, start_date=None, end_date=None):
        # Find boundaries of all time-series
        if start_date is None or end_date is None:
            start = pd.to_datetime(3162240000, unit='s')
            end = pd.to_datetime(0, unit='s')
            for ts in self.timeseries:
                t_start, t_end = ts.get_boundaries()
                if t_start < start:
                    start = t_start
                if t_end > end:
                    end = t_end
        if start_date is not None:
            start = pd.to_datetime(start_date)
        if end_date is not None:
            end = pd.to_datetime(end_date)

        self.batches = list()
        batch_duration = pd.Timedelta(batch_duration, unit='hours')
        b_start = start
        available_batches = 0
        dropped_batches = 0
        while b_start + batch_duration < end:
            batch = list()
            b_end = b_start + batch_duration
            missing = self.evaluate_missing(b_start, b_end)
            if not missing:
                for ts in self.timeseries:
                    batch.append(ts.get_time_window(b_start, b_end))
                self.batches.append(batch)
                available_batches = available_batches + 1
            else:
                dropped_batches = dropped_batches + 1
            b_start = b_start + pd.Timedelta(stride, unit='hours')

        print(str(available_batches) + ' usable batches, ' + str(dropped_batches) + ' dropped batches')
        return self.batches

    def omaelike_batcher(self, batch_duration, n_batches, stride=24*8, start_date=None, end_date=None):
        if start_date is None or end_date is None:
            start = pd.to_datetime(3162240000, unit='s')
            end = pd.to_datetime(0, unit='s')
            for ts in self.timeseries:
                t_start, t_end = ts.get_boundaries()
                if t_start < start:
                    start = t_start
                if t_end > end:
                    end = t_end
        if start_date is not None:
            start = pd.to_datetime(start_date)
        if end_date is not None:
            end = pd.to_datetime(end_date)

        self.batches = list()
        batch_duration = pd.Timedelta(batch_duration, unit='hours')
        available_batches = 0
        dropped_batches = 0
        b_start = start
        b_end = b_start+batch_duration
        while b_end < end:
            batch = list()
            missing = self.evaluate_missing(b_start, b_end)
            if not missing:
                for ts in self.timeseries:
                    batch.append(ts.get_time_window(b_start, b_end))
                self.batches.append(batch)
                available_batches = available_batches + 1
            else:
                dropped_batches = dropped_batches + 1

            b_start = b_start + pd.Timedelta(int(stride), unit='hours')
            b_end = b_start + batch_duration


        minutes = ((end - start).value - batch_duration.value)/60/1000/1000/1000
        while len(self.batches) < n_batches:
            batch = list()
            b_start = start + pd.Timedelta(int(np.random.randint(low=0, high=minutes, size=1)), unit='minutes')
            b_end = b_start + batch_duration
            missing = self.evaluate_missing(b_start, b_end)
            if not missing:
                for ts in self.timeseries:
                    batch.append(ts.get_time_window(b_start, b_end))
                self.batches.append(batch)
                available_batches = available_batches + 1
            else:
                dropped_batches = dropped_batches + 1

        print(str(available_batches) + ' usable batches, ' + str(dropped_batches) + ' dropped batches')
        return self.batches
    def create_n_batches(self, batch_duration, n_batches, start_date=None, end_date=None):
        if start_date is None or end_date is None:
            start = pd.to_datetime(3162240000, unit='s')
            end = pd.to_datetime(0, unit='s')
            for ts in self.timeseries:
                t_start, t_end = ts.get_boundaries()
                if t_start < start:
                    start = t_start
                if t_end > end:
                    end = t_end
        if start_date is not None:
            start = pd.to_datetime(start_date)
        if end_date is not None:
            end = pd.to_datetime(end_date)

        batch_duration = pd.Timedelta(batch_duration, unit='hours')
        minutes = ((end - start).value - batch_duration.value)/60/1000/1000/1000
        self.batches = list()
        available_batches = 0
        dropped_batches = 0
        while len(self.batches) < n_batches:
            batch = list()
            b_start = start + pd.Timedelta(int(np.random.randint(low=0, high=minutes, size=1)), unit='minutes')
            b_end = b_start + batch_duration
            missing = self.evaluate_missing(b_start, b_end)
            if not missing:
                for ts in self.timeseries:
                    batch.append(ts.get_time_window(b_start, b_end))
                self.batches.append(batch)
                available_batches = available_batches + 1
            else:
                dropped_batches = dropped_batches + 1

        print(str(available_batches) + ' usable batches, ' + str(dropped_batches) + ' dropped batches')
        return self.batches
    def evaluate_missing(self, start, end):
        for idx, ts in enumerate(self.timeseries):
            time = ts.get_time_window(start, end)[1]
            df = pd.DataFrame({'datetime': time})
            df['dt'] = df['datetime'].shift(-1) - df['datetime']
            maximum = df['dt'].max()
            if len(time)>0:
                end_gap = end - time[-1]
                start_gap = time[0] - start
                if self.missing_threshold[idx] is not None:
                    if not self.allow_missing[idx]:
                        if bool(maximum > self.missing_threshold[idx]) or end_gap > self.missing_threshold[idx] or \
                                start_gap > self.missing_threshold[idx]:
                            return True
            else:
                return True
        return False

    def return_single_batches(self, batch_duration, nbatches):
        if self.train_batches is not None:
            self.train_batches = nbatches

        start = pd.to_datetime(3162240000, unit='s')
        end = pd.to_datetime(0, unit='s')
        for ts in self.timeseries:
            t_start, t_end = ts.get_boundaries()
            if t_start < start:
                start = t_start
            if t_end > end:
                end = t_end

        train_b = self.create_n_batches(batch_duration=batch_duration, n_batches=nbatches, start_date=start,
                                              end_date=end)

        batches = BaseDataLoader(train_b, self.descriptions, self.is_predicted, self.inp_map, device=self.device, dtype=self.dtype, **self.batch_dict)

        return batches, self.inp_dim, self.out_dim, self.out_map, self.descriptions_map

    def split_train_val(self, batch_duration, val_per=0.2, single_training_batches=False, sequential_validation=False, sequential_stride=24, omaelike=False, warmup = pd.Timedelta(7, unit='days')):
        if self.train_batches is not None and self.val_batches is not None:

            start = pd.to_datetime(3162240000, unit='s')
            end = pd.to_datetime(0, unit='s')
            for ts in self.timeseries:
                t_start, t_end = ts.get_boundaries()
                if t_start < start:
                    start = t_start
                if t_end > end:
                    end = t_end
            train_duration = (1-val_per)*(end-start)
            val_start = start + train_duration - warmup

            if single_training_batches:
                batch = list()
                batches = list()
                for ts in self.timeseries:
                    batch.append(ts.get_time_window(start, end))

                batches.append((batch))
                train_b = batches
            elif omaelike:

                train_b = self.omaelike_batcher(batch_duration=batch_duration, n_batches=self.train_batches,
                                                stride=batch_duration,start_date=start,end_date=val_start)
            else:
                train_b = self.create_n_batches(batch_duration=batch_duration, n_batches=self.train_batches,
                                                start_date=start,
                                                end_date=val_start)

            train = BaseDataLoader(train_b, self.descriptions, self.is_predicted, self.inp_map,device=self.device, dtype=self.dtype, **self.batch_dict)

            if val_per > 0:
                if not sequential_validation:
                    validate_b = self.create_n_batches(batch_duration=batch_duration, n_batches=self.val_batches, start_date=val_start,
                                                          end_date=end)
                    validate = BaseDataLoader(validate_b, self.descriptions, self.is_predicted, self.inp_map,device=self.device,
                                              dtype=self.dtype, **self.batch_dict)
                else:
                    validate = self.sequential_batchs(batch_duration=batch_duration, stride=sequential_stride, start_date=val_start, end_date=end)
            else:
                validate = None

            return train, validate, self.inp_dim, self.out_dim, self.out_map, self.descriptions_map
        else:
            raise 'Must define train_batches and val_batches!'

    def sequential_batches_missing_before_forecast(self, missing_hours, forecast_window=24, batch_duration=8*24, stride=24, start_date=None, end_date=None, mdtp=None):
        start = pd.to_datetime(3162240000, unit='s')
        end = pd.to_datetime(0, unit='s')
        for ts in self.timeseries:
            t_start, t_end = ts.get_boundaries()
            if t_start < start:
                start = t_start
            if t_end > end:
                end = t_end
        if start_date is not None:
            start = start_date
        if end_date is not None:
            end = end_date

        batch_duration = pd.Timedelta(batch_duration, unit='hours')
        self.batches = list()
        available_batches = 0
        dropped_batches = 0
        b_start = start
        b_end = b_start+batch_duration
        while b_end < end:
            batch = list()
            missing = self.evaluate_missing(b_start, b_end)
            if not missing:
                for ts in self.timeseries:
                    if ts.is_forecast:
                        h = 0
                    else:
                        h = missing_hours
                    batch.append(ts.get_time_window_w_missing_before_forecast(b_start, b_end, forecast_window, h))
                self.batches.append(batch)
                available_batches = available_batches + 1
            else:
                dropped_batches = dropped_batches + 1

            b_start = b_start + pd.Timedelta(int(stride), unit='hours')
            b_end = b_start + batch_duration

        print(str(available_batches) + ' usable batches, ' + str(dropped_batches) + ' dropped batches')
        return BaseDataLoader(self.batches, self.descriptions, self.is_predicted, self.inp_map,device=self.device, dtype=self.dtype, **self.batch_dict)



    def sequential_batchs(self,batch_duration=8*24, stride=24, start_date=None, end_date=None, mdtp=None):
        start = pd.to_datetime(3162240000, unit='s')
        end = pd.to_datetime(0, unit='s')
        for ts in self.timeseries:
            t_start, t_end = ts.get_boundaries()
            if t_start < start:
                start = t_start
            if t_end > end:
                end = t_end
        if start_date is not None:
            start = start_date
        if end_date is not None:
            end = end_date

        batch_duration = pd.Timedelta(batch_duration, unit='hours')
        self.batches = list()
        available_batches = 0
        dropped_batches = 0
        b_start = start
        b_end = b_start+batch_duration
        while b_end < end:
            batch = list()
            missing = self.evaluate_missing(b_start, b_end)
            if not missing:
                for ts in self.timeseries:
                    if ts.is_forecast:
                        m = None
                    else:
                        m = mdtp
                    batch.append(ts.get_time_window(b_start, b_end, mdtp=m))
                self.batches.append(batch)
                available_batches = available_batches + 1
            else:
                dropped_batches = dropped_batches + 1

            b_start = b_start + pd.Timedelta(int(stride), unit='hours')
            b_end = b_start + batch_duration

        print(str(available_batches) + ' usable batches, ' + str(dropped_batches) + ' dropped batches')
        return BaseDataLoader(self.batches, self.descriptions, self.is_predicted, self.inp_map,device=self.device, dtype=self.dtype, **self.batch_dict)

    def split_k_folds(self, n_folds, batch_duration, stride):
        train = list()
        validate = list()
        splits = list()
        all_idx = range(0, n_folds)

        start = pd.to_datetime(3162240000, unit='s')
        end = pd.to_datetime(0, unit='s')
        for ts in self.timeseries:
            t_start, t_end = ts.get_boundaries()
            if t_start < start:
                start = t_start
            if t_end > end:
                end = t_end
        fold_duration = (end-start)/n_folds
        dates = list()
        for idx in range(n_folds+1):
            dates.append(start + fold_duration*idx)

        for idx in range(n_folds):
            splits.append(self.create_batches(batch_duration=batch_duration,stride=stride, start_date=dates[idx],
                                                  end_date=dates[idx + 1]))
        for idx in range(n_folds):
            train.append(BaseDataLoader(list(itertools.chain.from_iterable(list(splits[0:idx]+splits[idx+1:]))), self.descriptions, self.is_predicted, self.inp_map,device=self.device, dtype=self.dtype, **self.batch_dict))
            validate.append(BaseDataLoader(splits[idx], self.descriptions, self.is_predicted, self.inp_map,device=self.device, dtype=self.dtype, **self.batch_dict))

        return train, validate, self.inp_dim, self.out_dim, self.out_map, self.descriptions_map

    def split_kn_folds_singles_batches_training(self, n_folds, batch_duration):
        train = list()
        validate = list()
        val_splits = list()
        train_splits = list()
        all_idx = range(0, n_folds)
        start = pd.to_datetime(3162240000, unit='s')
        end = pd.to_datetime(0, unit='s')
        for ts in self.timeseries:
            t_start, t_end = ts.get_boundaries()
            if t_start < start:
                start = t_start
            if t_end > end:
                end = t_end
        fold_duration = (end-start)/n_folds
        dates = list()
        for idx in range(n_folds+1):
            dates.append(start + fold_duration*idx)

        for idx in range(n_folds):
            val_splits.append(self.create_n_batches(batch_duration=batch_duration,n_batches=self.val_batches, start_date=dates[idx],
                                                   end_date=dates[idx + 1]))
            batch = list()
            n_batch = list()
            for ts in self.timeseries:
                batch.append(ts.get_time_window(dates[idx], dates[idx+1]))
            n_batch.append(batch)
            train_splits.append(n_batch)

        for idx in range(n_folds):
            train.append(BaseDataLoader(list(itertools.chain.from_iterable(list(train_splits[0:idx]+train_splits[idx+1:]))), self.descriptions, self.is_predicted, self.inp_map,device=self.device, dtype=self.dtype, **self.batch_dict))
            validate.append(BaseDataLoader(val_splits[idx], self.descriptions, self.is_predicted, self.inp_map,device=self.device, dtype=self.dtype, **self.batch_dict))

        return train, validate, self.inp_dim, self.out_dim, self.out_map, self.descriptions_map
    def split_kn_folds(self, n_folds, batch_duration):
        train = list()
        validate = list()
        val_splits = list()
        train_splits = list()
        all_idx = range(0, n_folds)

        start = pd.to_datetime(3162240000, unit='s')
        end = pd.to_datetime(0, unit='s')
        for ts in self.timeseries:
            t_start, t_end = ts.get_boundaries()
            if t_start < start:
                start = t_start
            if t_end > end:
                end = t_end
        fold_duration = (end-start)/n_folds
        dates = list()
        for idx in range(n_folds+1):
            dates.append(start + fold_duration*idx)

        for idx in range(n_folds):
            val_splits.append(self.create_n_batches(batch_duration=batch_duration,n_batches=self.val_batches, start_date=dates[idx],
                                                   end_date=dates[idx + 1]))
            train_splits.append(self.create_n_batches(batch_duration=batch_duration,n_batches=round(self.train_batches/(n_folds-1)), start_date=dates[idx],
                                                   end_date=dates[idx + 1]))


        for idx in range(n_folds):
            train.append(BaseDataLoader(list(itertools.chain.from_iterable(list(train_splits[0:idx]+train_splits[idx+1:]))), self.descriptions, self.is_predicted, self.inp_map,device=self.device, dtype=self.dtype, **self.batch_dict))
            validate.append(BaseDataLoader(val_splits[idx], self.descriptions, self.is_predicted, self.inp_map,device=self.device, dtype=self.dtype, **self.batch_dict))

        return train, validate, self.inp_dim, self.out_dim, self.out_map, self.descriptions_map

    def __iter__(self):
        self.idx_iter = -1
        return self

    def __next__(self):
        self.idx_iter = self.idx_iter + 1
        if self.idx_iter < len(self.batches):
            return self.batches[self.idx_iter], self.batches[self.idx_iter]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.batches)
