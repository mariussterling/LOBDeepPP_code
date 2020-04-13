#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:26:54 2019

@author: ms
"""

import os
if __name__ == '__main__':
    import platform
    if platform.node() == 'mstp':
        os.chdir('/home/ms/github/ob_nw')

import numpy as np
from LOBDeepPP.LOBDeepPP_class import LOB
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence


class tsg_wrapper(Sequence):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'lag' not in self.kwargs:
            self.kwargs.update({'lag': 3})
        if 'pred_horizon' not in self.kwargs:
            self.kwargs.update({'pred_horizon': [1]})
        if 'levels' not in self.kwargs:
            self.kwargs.update({'levels': 100})
        if 'batch_size' not in self.kwargs:
            self.kwargs.update({'batch_size': 128})
        self.files = self.kwargs.get('files', None)
        self.set_lob(self.files[0])
        self.set_tsg()

    def set_tsg(self):
        data, targets, time_data, time_targets = self.lob.get_data_targets(
            **self.kwargs)
        self.tsg = TimeseriesGenerator_ob(
            data=data,
            targets=targets,
            length=self.kwargs.get('lag'),
            time_data=time_data,
            time_targets=time_targets,
            **self.kwargs)

    def get_index(self, index):
        self.set_lob_data_targets_by_index(index)
        return index % len(self.tsg)

    def set_lob(self, file):
        self.lob = LOB(file_path=file)
        self.lob.transform()
        self.lob.adjust_time()

    def __len__(self):
        return len(self.files) * len(self.tsg)

    def set_lob_data_targets_by_index(self, index):
        file_number = int(index // len(self.tsg))
        if self.files[file_number] != self.lob.file_path:
            self.set_lob(self.files[file_number])
            self.tsg.data, self.tsg.targets, self.tsg.time_data,\
                self.tsg.time_targets = self.lob.get_data_targets(
                    **self.kwargs)

    def __getitem__(self, index):
        return self.tsg[self.get_index(index)]


class TimeseriesGenerator_ob(TimeseriesGenerator):
    def __init__(self, *,
                 data,
                 targets,
                 length,
                 stride=1,
                 start_index=0,
                 end_index=None,
                 shuffle=False,
                 reverse=False,
                 batch_size=128,
                 stack_samples=False,
                 add_channel=False,
                 fillna=None,
                 **kwargs):
        self.kwargs = kwargs
        if len(data) != len(targets):
            raise ValueError(
                f'Data and targets have to be of same length. Data length is '
                f'{len(data)} while target length is {len(targets)}')

        self.data = data
        self.targets = targets
        self.time_data = kwargs.get('time_data', None)
        self.time_targets = kwargs.get('time_targets', None)

        if isinstance(length, list):
            self.length = length
        elif isinstance(length, int):
            self.length = [i for i in range(length)]
        else:
            ValueError('length must be integer or list of integer')

        self.stride = stride
        self.start_index = start_index + max(self.length)
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.stack_samples = stack_samples
        self.add_channel = add_channel
        self.fillna = fillna

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

    def __getitem__(self, index):
        if self.shuffle:
            DeprecationWarning(
                'shuffle is not yet implemented in the TimeseriesGenerator_ob')
            # FIXME:NotImplementedError
        else:
            if self.batch_size is not None:
                i = self.start_index + self.batch_size * self.stride * index
                rows = np.arange(i, min(i + self.batch_size * self.stride,
                                        self.end_index + 1), self.stride)
            else:
                rows = np.arange(self.start_index, self.end_index + 1,
                                 self.stride)

        samples = np.array([self.data[row - np.array(sorted(
            self.length, reverse=not self.reverse))] for row in rows])
        targets = np.array([self.targets[row] for row in rows])
        time_samples = None
        time_targets = None

        if self.stack_samples:
            samples = self.stack_orders(samples)
        if self.fillna is not None:
            samples[np.isnan(samples)] = self.fillna
        if self.add_channel:
            samples = samples[..., np.newaxis]
        if self.kwargs.get('targets_standardize_by_sqrt_time', False):
            targets /= 1 / 10 * np.arange(1, targets.shape[1] + 1).reshape(
                [1, targets.shape[1], 1]) ** 0.5
        if self.kwargs.get('targets_stacked', True):
            targets = self.stack_targets(targets)
        if self.time_data is not None and self.time_targets is not None:
            time_samples = np.array([self.time_data[row - np.array(sorted(
                self.length, reverse=not self.reverse))] for row in rows])
            time_targets = self.stack_targets(self.time_targets[rows])
            return samples, targets, time_samples, time_targets
        return samples, targets

    def stack_orders(self, samples):
        dims = samples.shape
        return samples.reshape([dims[0], dims[1], dims[2] * dims[3]])

    def unstack_orders(self, samples):
        dims = samples.shape
        return samples.reshape(dims[0], dims[1], dims[2] // 2, 2)

    def stack_targets(self, targets):
        return targets.reshape([-1, targets.shape[1] * 2])

    def unstack_targets(self, targets):
        return targets.reshape([-1, targets.shape[1] // 2, 2])

    def __len__(self):
        if self.batch_size is not None:
            return (self.end_index - self.start_index + self.batch_size
                    * self.stride) // (self.batch_size * self.stride)
        else:
            return 1
