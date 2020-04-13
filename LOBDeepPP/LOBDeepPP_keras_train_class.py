import os
import pickle
import json
import copy
import numpy as np
from datetime import datetime
from multiprocessing import cpu_count
from keras import layers, models, callbacks

from LOBDeepPP.LOBDeepPP_model import LOBDeepPP_models
from .LOBDeepPP_class.LOBDeepPP_DataGenerator import tsg_wrapper
from .LOBDeepPP_keras_train_class_functions import set_model_weights
from .LOBDeepPP_model.__activations import PReLU2, LeakyReLU2
from .LOBDeepPP_model.__metric import mde_perc


class LOBDeepPP_keras_train_class:
    def __init__(self, lob_model):
        """
        Constructor of the keras model.

        lob_model : str
            Name of keras model.
        """
        self.__set_model(lob_model)

    def __set_model(self, lob_model):
        """
        Setter for model name.

        lob_model : str
            Name of keras model.
        """
        poss_models = [i for i in dir(LOBDeepPP_models) if 'lob_keras_model' in
                       i.lower()]
        self.lob_model = lob_model
        if 'LOBDeepPP_keras_model' + self.lob_model in poss_models:
            self.lob_model = 'LOBDeepPP_keras_model' + self.lob_model
        if self.lob_model not in poss_models:
            raise AttributeError(
                '\n'.join(['Only these models are allowed in lob_model: \
                           \n- {:s}'.format('\n- '.join(poss_models))
                           ]))

    def set_model_params(self, file_path=None):
        """
        Set parameters from file.

        Parameters
        ----------
        file_path : str
            Path to the parameters file.
        """
        if file_path is not None:
            self.model_params_file_path = file_path
            with open(file_path, 'r') as f:
                self.model_params = json.load(f)
            self.model_params['date_span'] = [
                np.datetime64(i) for i in self.model_params.get('date_span')]
        else:
            raise NotImplementedError('LOBDeepPP_keras_train_class.py'
                                      + '::set_model_params')

    def set_gen(self, train=None, valid=None, test=None):
        """
        Setting training, validation and testing data set.

        Parameters
        ----------
        train, valid, test : list of file paths
            List of paths to training, validation and testing data.
        """
        self.gen = {}
        if train is not None:
            self.gen.update({'train':
                             tsg_wrapper(
                                 files=train,
                                 **self.model_params['lob_model'])})
        if valid is not None:
            self.gen.update({'valid':
                             tsg_wrapper(
                                 files=valid,
                                 **self.model_params['lob_model'])})
        if test is not None:
            self.gen.update({'test':
                             tsg_wrapper(
                                 files=test,
                                 **self.model_params['lob_model'])})
        assert len(self.gen) > 0, 'set_gen: no generators are set!'

    def set_model(self, **kwargs):
        """
        Creating keras model based on arguments from file.

        Parameters
        ----------
        kwargs :
            Keyword arguments passed to the defined model.
        """
        input_shape = self.gen['train'][0][0].shape[1:]
        output_shape = self.gen['train'][0][1].shape[1:]
        inp = layers.Input(shape=input_shape, name='input')
        if 'params' in kwargs.keys():
            params = kwargs.pop('params')
        else:
            params = self.model_params

        self.model = getattr(LOBDeepPP_models, self.lob_model)(
            inp=inp,
            output_shape=list(map(int, np.array(output_shape).squeeze())),
            params=params,
            **kwargs
        )

    def set_file_paths(self, path_data='Data/Matlab', override=False):
        """
        Sets the file paths in the model params according to the provided
        date span, if the do not yet exist or override is true.

        Parameters
        ----------
        path_data : str, optional
            Path to the matlab data files. The default is 'Data/Matlab'.
        override : bool, optional
            Override existing parameter options. The default is False.

        Returns
        -------
        None.

        """
        # Checks whether model file paths are already set.
        if (override
            or (self.model_params.get('lob_model_file_paths', None)
                is not None)):
            return

        # Subsetting existing files in the directory
        file_paths = [os.path.join(path_data, i) for i in
                      sorted(os.listdir(path_data))]
        file_paths_dict = {i: np.datetime64(i.split('/')[-1].rstrip('.mat'))
                           for i in file_paths}
        file_paths_dict = {k: v for k, v in file_paths_dict.items() if
                           (v >= self.model_params['date_span'][0]
                            and v <= self.model_params['date_span'][1]
                            )}

        # Loading the data: train, val, test data split daywise:
        #   - Train ~50%, with >50%
        #   - Val   ~25%, with >25%
        #   - Test  ~25%, with <25%
        n = len(file_paths_dict)
        splits_at = [int(np.ceil(n / 2)), int(np.ceil(n * 3 / 4))]
        file_paths_list = [k for k, v in file_paths_dict.items()]
        self.model_params.update({
            'lob_model_file_paths': {
                'train': file_paths_list[:splits_at[0]],
                'valid': file_paths_list[:splits_at[0]:splits_at[1]],
                'test': file_paths_list[splits_at[1]:]}})

    def set_up_model(self, **kwargs):
        def __get_path(k):
            if '*' not in k:
                return k
            files = os.listdir(os.path.dirname(k))
            files = sorted(
                [i for i in files if all([
                    j in i for j in
                    os.path.basename(k).split('*')])])
            return os.path.join(os.path.dirname(k), files[-1])
        if self.model_params.get('load_model', None) is not None:
            self.load_model(__get_path(self.model_params.get('load_model')))
        elif not hasattr(self, 'model'):
            self.set_up_new_model(**kwargs)
        if self.model_params.get('set_model_weights', None) is not None:
            tmp = LOBDeepPP_keras_train_class(self.lob_model)
            tmp.load_model(__get_path(
                self.model_params.get('set_model_weights')))
            self.set_model_weights(tmp.model)
            del tmp
        if self.model_params.get('set_submodel_weights', None) is not None:
            for k, v in self.model_params.get('set_submodel_weights').items():
                tmp = LOBDeepPP_keras_train_class(
                    os.path.basename(os.path.dirname(k)))
                tmp.load_model(__get_path(k))
                for i in v:
                    self.set_submodel_weights(tmp.model, layer_name=i)
                del tmp

        if self.model_params['tcn']['base'].get('dropout_rate', None) \
                is not None:
            for l in self.model.get_layer('tcn_base').layers:
                if isinstance(l, layers.SpatialDropout2D):
                    l.rate = self.model_params['tcn']['base']['dropout_rate']

    def set_up_new_model(
            self,
            PATH_NN_MODEL='results/nn_model_training',
            start_time=datetime.now().strftime('%Y%m%d_%H%M%S'),
            SEED=42,
            **kwargs):

        self.start_time = start_time
        self.path_nn_model = PATH_NN_MODEL
        del start_time, PATH_NN_MODEL

        # set data files and model
        self.set_file_paths()
        self.set_gen(**self.model_params['lob_model_file_paths'])
        self.set_model(**kwargs)

        # crete directory to save results
        try:
            os.makedirs(os.path.join(self.path_nn_model, self.model.name))
        except FileExistsError:
            pass
        # save initial model weights
        fn1 = os.path.basename(self.model_params_file_path).replace('.json',
                                                                    '')
        fn = f'{fn1}___{self.start_time}___model_00.h5'
        self.model.save(os.path.join(self.path_nn_model, self.model.name, fn))
        del fn
        self.initial_epochs = 0
        # save model parameters
        fn = f'{fn1}___{self.start_time}___model_params.json'
        file = os.path.join(self.path_nn_model, self.model.name, fn)
        with open(file, 'w') as outfile:
            mp = copy.deepcopy(self.model_params)
            json.dump(mp, outfile, default=str)
        del fn, file

    def get_latest_model(self, path='results/nn_model_training',
                         lob_model=None):
        if lob_model is None:
            lob_model = sorted(os.listdir(path))[-1]
        files = sorted([i for i in os.listdir(os.path.join(path, lob_model)) if
                        '_model_' in i and '.h5' in i])
        if len(files) == 0:
            return None
        return os.path.join(path, lob_model, files[-1])

    def load_model(self, path):
        self.path_nn_model = os.path.dirname(os.path.dirname(path))
        lob_model = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        self.start_time = filename.split('___')[1]
        self.initial_epochs = int(
            filename.split('___')[-1].replace('weights_', '')
                                     .replace('model_', '')
                                     .replace('.hdf5', '')
                                     .replace('.h5', '')
                                     .replace('_TEST', ''))
        files = os.listdir(os.path.join(self.path_nn_model, lob_model))
        files = [i for i in files if self.start_time in i
                 and filename.split('___')[0] in i and '.h5' in i]

        # read in model_params
        if not hasattr(self, 'model_params'):
            fn = os.path.join(
                self.path_nn_model,
                lob_model,
                '___'.join(filename.split('___')[:2] + ['model_params.json']))
            self.set_model_params(fn)
            del fn

        if self.model_params.get('lob_model_file_paths', None) is None:
            self.set_file_paths()
        # create generators
        if not hasattr(self, 'gen'):
            self.set_gen(**self.model_params['lob_model_file_paths'])
        # create model
        if 'weights' in filename and 'h5fs' in filename:
            self.set_model(lob_model)
            # load model weights
            self.model.load_weights(path)
        else:
            # load model
            self.model = models.load_model(
                path,
                custom_objects={
                    'NonPos': LOBDeepPP_models.NonPos,
                    'PReLU2': PReLU2,
                    'PReLU': PReLU2,
                    'LeakyReLU2': LeakyReLU2,
                    'LeakyReLU': LeakyReLU2
                },
                compile=False)
        self.model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['logcosh', 'mean_absolute_percentage_error', mde_perc])

    def get_callbacks(self, opt):
        # ModelCheckpoints: saving model after each epoch
        fn1 = (os.path.basename(self.model_params_file_path)
               .replace('.json', ''))
        fn = (f'{fn1}___{self.start_time}'
              f'___model_%s{"_TEST" if opt.test else ""}.h5' % ('{epoch:02d}'))
        filepath = os.path.join(self.path_nn_model, self.model.name, fn)
        del fn
        checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                               verbose=opt.verbose)
        # TerminateOnNaN
        tonan = callbacks.TerminateOnNaN()
        # History
        history = callbacks.History()
        # CSV logger: saves epoch train and valid loss to a log file
        fn1 = (os.path.basename(self.model_params_file_path)
               .replace('.json', ''))
        fn = (f'{fn1}___{self.start_time}'
              f'___training{"_TEST" if opt.test else ""}.log')
        filepath = os.path.join(self.path_nn_model, self.model.name, fn)
        csv_logger = callbacks.CSVLogger(filepath, separator=',', append=True)
        # Learning rate scheduler

        def exp_decay(
                epoch,
                initial_lrate=self.model_params['keras_train']['lr'],
                decay=self.model_params['keras_train']['lr_decay']):
            lrate = initial_lrate * np.exp(-decay * epoch)
            return lrate

        def learning_rate_decay(
                epoch,
                initial_lrate=self.model_params['keras_train']['lr'],
                decay=self.model_params['keras_train']['lr_decay']):
            lrate = initial_lrate * (1 - decay) ** epoch
            return lrate
        lrs = callbacks.LearningRateScheduler(learning_rate_decay)
        callbacks_list = [tonan, checkpoint, history, csv_logger, csv_logger,
                          lrs]
        # Early stopping: stops training if validation loss does not improves
        if (self.model_params['keras_train'].get('early_stopping_n') is not
                None):
            es = callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=self.model_params['keras_train']['early_stopping_n'],
                verbose=opt.verbose)
            callbacks_list.append(es)
        return callbacks_list

    def train(self, opt, **kwargs):
        if self.model_params['keras_train']['epochs'] is None:
            epochs = self.initial_epochs + (2 if opt.test else 5)
        else:
            epochs = (self.initial_epochs
                      + self.model_params['keras_train']['epochs'])

        h = self.model.fit_generator(
            epochs=epochs,
            initial_epoch=self.initial_epochs,
            generator=self.gen['train'],
            validation_data=self.gen['valid'],
            callbacks=self.get_callbacks(opt),
            # nb_val_samples=4,#len(training_generator),
            validation_steps=4 if opt.test else len(self.gen['valid']),
            steps_per_epoch=4 if opt.test else len(self.gen['train']),
            use_multiprocessing=True,
            workers=max(1, (cpu_count() * 90) // 100),
        )
        fn1 = (os.path.basename(self.model_params_file_path)
               .replace('.json', ''))
        fn = (f'{fn1}___{self.start_time}'
              f'___history___'
              f'{int(self.initial_epochs):02d}_{int(epochs):02d}'
              f'{"_TEST" if opt.test else ""}.pkl')
        file = os.path.join(self.path_nn_model, self.model.name, fn)
        with open(file, 'wb') as f:
            pickle.dump(h, f)
        self.initial_epochs = epochs

    def set_model_weights(self, model):
        self.model = set_model_weights(self.model, model)

    def set_submodel_weights(self, model, layer_name=None):
        if layer_name is None:
            layer_name = model.name
        self.model.get_layer(layer_name).set_weights(
            model.get_layer(layer_name).get_weights())
