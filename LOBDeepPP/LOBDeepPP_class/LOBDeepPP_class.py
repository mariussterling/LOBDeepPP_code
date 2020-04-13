import scipy.io
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from collections import Counter, OrderedDict
from scipy.stats.mstats import gmean, hmean
from scipy.stats import binned_statistic


class LOB:
    """
    Limit Order Book class for Matlab (Mat) files containing book and time
    information.

    Parameters
    ----------
    file_path : str
        Name of the mat file.
    """
    def __init__(self, *, file_path=None, ob=None, time=None):
        """
        Initialization of LOB class.

        Parameters
        ----------
        file_path : str
            Path of the mat file.
        ob : numpy.array
            Order book.
        time : numpy.array
            Time corresponding to the order book.
        """
        if file_path is not None:
            self.file_path = file_path
            tmp = scipy.io.loadmat(file_path)
            self.__ob = tmp['book']  # .reshape([tmp['book'].shape[0],400,2])
            self.__time = tmp['time']
            self.transform()
            self.__ob[(self.__ob[:, :, 1] == 0)] = np.nan
        elif ob is not None and time is not None:
            self.file_path = None
            assert ob.shape[0] == time.shape[0], 'ob and time must have same \
                                                  length in first dimension!'
            self.__ob = ob
            self.__time = time
        else:
            raise ValueError("Either 'file_path' or 'ob' and 'time' must be \
                             provided.")
    # ==========================================================================
    #      Getter/setter accessibility for ob and time:
    #          - public getter ("self.ob")
    #          - private setter ("self.__ob = ...")
    # ==========================================================================

    def __get_ob(self):
        """Private getter fucntion for orderbook ob"""
        return self.__ob
    ob = property(__get_ob)

    def __get_time(self):
        """Private getter fucntion for time"""
        return self.__time
    time = property(__get_time, None)

    def adjust_time(self):
        """
        Adjusting time and transforming dtype into datetime.
        """
        # FIXME: raise DeprecationWarning() as soon as datetime is correct
        if all([isinstance(i, datetime) for i in self.time]):
            return
        dtype = list(set(i.__class__ for i in self.time))
        assert len(dtype) == 1, 'multiple dtypes in self.time'
        if dtype[0] is datetime:
            return

        def __adjust_time_one(matlab_datenum):
            return datetime.fromordinal(int(matlab_datenum))\
                   + timedelta(days=matlab_datenum % 1)\
                   - timedelta(days=366)
            # microseconds=((matlab_datenum%1*(24*60*60))%1)*10**6)
        self.__time = [__adjust_time_one(i[0]) for i in self.time]

    def time_adjustment(self):
        time = np.vectorize(
                lambda t: t.replace(microsecond=0) + timedelta(seconds=1)
                )(self.time)
        diff = np.vectorize(lambda t: t.total_seconds())(np.diff(time, axis=0))
        if (diff == -1).all() or (diff == 1).all():
            return
        elif (diff < 0).any() and (diff > 0).any():
            raise NotImplementedError('The time is neither increasing nor\
                                      decreasing!!!')
        else:
            pass
        delta = np.vectorize(
                lambda t: timedelta(seconds=1) if t == 0 else
                timedelta(seconds=-1))(np.argmin(time, axis=0))
        time = time[0] + np.array([i*delta for i in range(time.shape[0])])
        self.__time = time[:]

    def check_date_correct(self):
        raise DeprecationWarning()
        day = datetime.strptime(os.path.basename(self.file_path).replace(
                '.mat', ''), '%Y-%m-%d').date()
        self.adjust_time()
        return all([i.date() == day for i in self.time])

    def check_intraday_complete(self, verbose=True,
                                return_missing_duplicates=False):
        raise DeprecationWarning()
        complete = True
        date_wo_ms = [i.replace(microsecond=0) for i in self.time]
        border = [date_wo_ms[0].replace(hour=9, minute=30, second=0,
                                        microsecond=0),
                  date_wo_ms[0].replace(hour=16, minute=0, second=0,
                                        microsecond=0)]
        time_interval = int((border[1]-border[0]).total_seconds())
        date_unique = list(np.unique(date_wo_ms))
        date_missing = sorted(set(border[0]+timedelta(seconds=i) for i in range(time_interval+1)) - set(date_unique))
        if date_missing:
            if verbose: print('Missing date points: %i'%len(date_missing))
            complete = False
        duplicates = {i[0]:i[1] for i in Counter(date_wo_ms).items() if i[1]!=1}
        if duplicates:
            if verbose: print('Missing date points: %i'%len(duplicates))
            complete = False
        if return_missing_duplicates:
            for i in date_missing:
                duplicates.update({i:0})
            return complete, OrderedDict(sorted(duplicates.items()))
        else:
            return complete
    def get_mean_price(self,*,method='mean',level=1):
        """
        Computes wighted mean prices of the order book
        !!!#TODO
        """
        if method == 'volume weighted':
            tmp = self.ob[:,:(level*2),0] * self.ob[:,:(level*2),1]
            return tmp.sum(axis=1)/self.ob[:,:(level*2),1].sum(axis=1)
        elif method == 'mean':
            return self.ob[:,:(level*2),0].mean(axis=1)
        elif method in ['geometric mean','gmean']:
            return gmean(self.ob[:,:(level*2),0],axis=1)
        elif method in ['harmonic mean','hmean']:
            return hmean(self.ob[:,:(level*2),0],axis=1)
        else:
            raise ValueError('method needs to be valid string see "LOB.computer_mean_price.__doc__()"')
    def get_returns(self,*,mean_prices=None,log_returns=True,return_time=False):
        """
        Computes the (log) returns of the prices.
        
        Parameters
        ----------
        mean_prices : numpy.array, optional
            Output of get_mean_price.
            Default: None, mean_price = self.get_mean_price()
        log_returns : bool, optiona;
            If True computes log returns, else returns.
            Default: True
        return_time: bool, optional
            If True function returns tuple, including the time.
            Default: False
        
        Returns
        -------
        (ret_ob, time) : tuple
            Containing ret_ob and time
        ret_ob : numpy.array 
            orderbook containing (log) returns. 
            One less observation, see np.array(self.ob.shape) - np.array(ret_ob[1:].shape) = [1,0,0]
        time : numpy.array
            Adjusted time variable for the returns
            One less observation, see len(self.time) - len(time) = 1
        """
        ret_ob = np.array(self.ob,copy=True)
        if mean_prices is None:
            mean_prices = self.get_mean_price()
        if log_returns:
            ret_ob[1:,:,0] = np.log(ret_ob[1:,:,0]) - np.log(mean_prices)[:-1,np.newaxis]
        else:
            ret_ob[1:,:,0] /= mean_prices[:-1,np.newaxis]
            ret_ob[1:,:,0] -= 1
        return ret_ob[1:], self.time[1:]
        
    def get_centered_prices(self,ob=None,*,mean_prices=None):
        if mean_prices is None:
            mean_prices = self.get_mean_price()
        if ob is None:
            ret_ob = np.array(self.ob,copy=True)
        else:
            ret_ob = ob
        ret_ob[:,:,0] -= mean_prices[:,np.newaxis]
        return ret_ob
    def standardise_prices(self,ob=None,*,sigma):
        #sigma = mean_prices *0 + mean_prices.var()**0.5
        if ob is None:
            ret_ob = np.array(self.ob,copy=True)
        else:
            ret_ob = ob
        ret_ob[:,:,0] /= sigma[:,np.newaxis]
        return ret_ob
    def normalize_prices(self,center_method='mean',center=True,standardize=True):
        #TODO!!!
        raise DeprecationWarning('Do not use this')
        if not center and not standardize:
            return
        prices = self.get_mean_price(method=center_method)
        if center:
            self.center_prices(prices)
        if standardize:
            sigma = prices *0 + prices.var()**0.5
            h = 15*60
            
            #for h in [i*60 for i in [1,5,15,30,60,120,180,240]]:
            for h in [int(i*60) for i in [2,30,60*4]]:
                sigma = np.empty([h,1])*0 + 0.0001
                sigma = np.append(sigma,np.array([prices[i:i+h].var()**0.5 for i in range(len(prices)-h)]))
                plt.plot(sigma,label = h//60)
                del sigma
            plt.legend()
            sigma = ((prices**2).cumsum() / np.arange(1,len(prices)+1))**0.5
            self.standardise_prices(sigma)
    def transform(self):
        if self.ob.ndim == 2:
            s = self.__ob.shape
            self.__ob = self.__ob.reshape([s[0],s[1]//2,2])
    def inverse_transform(self):
        if self.ob.ndim == 3:
            s = self.ob.shape
            self.__ob = self.__ob.reshape([s[0],s[1]*2])
    def get_data_targets(self,*,lag=3,pred_horizon=[5],levels=100,
                         log_return_method = None,return_time=False,**kwargs):
        """
        Returns feature and target variables of LOB data.
        
        Parameters
        ----------
        lag : int , list(int)
            Lagged time steps included. 
            Default: [0,1,2,3]
        pred_horizon : list(ist)
            Prediction horizon (time steps) for the target variable. 
            Default: [5]
        levels : int
            Level up to which orders are included.
            Default: 100
        idx : int
            Index for batch start.
            Default: 0
        batch_size : int or None, optional
            Size of the batch. If None all time points after idx.
            Default: None
        log_return_method : !!!#TODO
        
        return_time: bool
            Returns the times for data and targets.
            Default: False
        Returns 
        -------
        X,y : tuple
            Feature variable and target variable are returned, if return_time=False (default).
        X,y,time_x,time_y : tuple
            Feature variable and target variable plus times are returned, if return_time=True.
        """
        def normalize_all(x,mu=None,std=None):
            #FIXME
            if mu is None:
                mu = np.nanmean(x,axis=0)[np.newaxis]
            if std is None:
                sd = np.nanstd(x,axis=0)[np.newaxis]
                sd[sd == 0] = 1
            return (x - mu)/sd
        def scaler_all(x,ql=None,qu=None,quantile=[0.001,0.999],range_=[0,1]):
            if ql is None:
                ql = np.nanquantile(x,quantile[0],axis=0)[np.newaxis]
            if qu is None:
                qu = np.nanquantile(x,quantile[1],axis=0)[np.newaxis]
            d = qu-ql
            d[d == 0 ] = 1
            x = (x - ql) / d
            return x * (range_[1] - range_[0]) + range_[0]
        def normalize_first(x,mu=None,std=None):
            #FIXME
            if mu is None:
                mu = np.tile(np.nanmean(x[:,:2],axis=0),(x.shape[1]//2))[np.newaxis]
            if std is None:
                std = np.tile(np.nanstd(x[:,:2],axis=0),(x.shape[1]//2))[np.newaxis]
                std[std == 0] = 1
            return (x - mu)/std
        if False:
            ob[:,10:12]
            plt.hist(normalize_all(ob[...,0]).flatten(),bins=50,range=[-5,5])
            x = scaler_all(ob[...,1],range_=[-1,1],quantile=[0.01,0.99]).flatten()
            plt.hist(x[~np.isnan(x)],bins=50,range=[-1.5,1.5])
        
        ob = np.array(self.ob[:,:levels*2],copy=True)
        if isinstance(lag,int):
            lag = [i for i in range(lag+1)]
        elif isinstance(lag,list):
            pass
        else:
            raise ValueError('lag needs to be int or list')
        method_allowed = {
                'level_1_prices' : 'Computing log returns based on the previous periods level 1 bid and ask price',
                'same_level_prices': 'computing log return based on the previous period wrt the same level ',
                'None' : 'no logarithmatization or differentiation'
                }
        m1 = max(pred_horizon)
        m2 = ob.shape[0] - m1
        if log_return_method is None:
            target = np.array([ob[[t+i for i in pred_horizon]][:,:2,0] for t in range(0, m2)])
            #if kwargs.get('log_base_volume',None) is not None:
            #    ob[:,:,1] = np.log(1+ob[:,:,1])/np.log(kwargs.get('log_base_volume'))
            
        else:
            # target variables as log returns
            x = [(np.log(ob[h:(m2+h),:2,0]) - np.log(ob[:(m2),:2,0]))[:,np.newaxis] for h in pred_horizon]
            target= np.concatenate(x,axis=1)
            # ob as log_returns
            if log_return_method  == 'level_1_prices':
                #ob[:,:2,0].std()
                ob[1:,::2,0] = np.log(ob[1:,::2,0]) - np.log(ob[:-1,0,0])[:,np.newaxis]
                ob[1:,1::2,0] = np.log(ob[1:,1::2,0]) - np.log(ob[:-1,1,0])[:,np.newaxis]
                ob[0,~np.isnan(ob[0,:,0]),0] = 0
                
                ob[1:,:,0] *= 10**3 #TODO: Include Have proper moving std model!
                target *= 10**3 #TODO: Include Have proper moving std model!
                if False:
                    obc = np.cumsum(ob[:,:2,0],axis=0)
                    x = [(obc[h:] - obc[:-h])[:m2,np.newaxis]  for h in pred_horizon]
                    target2= np.concatenate(x,axis=1)
                    np.equal(target,target2).all()
                    #scaler_all(ob[:,:,0])
            elif log_return_method  == 'same_level_prices':
                ob[1:,:,0] = np.diff(np.log(ob[:,:,0]),axis=0)
                ob[0,~np.isnan(ob[0,:,0]),0] = 0
                # same result as: np.log(ob[1:,:,0]) - np.log(ob[:-1,:,0])
            else:
                opt = '\n- '.join(["'%s' : %s"%(k,v) for k,v in method_allowed.items()])
                raise ValueError("Only the following options are allowed for argument 'log_return_method':\n- %s"%(opt))
        if kwargs.get('log_base_volume',None) is not None:
            ob[:,:,1] = np.log(ob[:,:,1])/np.log(kwargs.get('log_base_volume'))
        if False:
            l = ob.shape[1]//4
            mu = np.nanmean(ob[:,:,0],axis=0)
            mu[::2] = mu[2*l]
            mu[1::2] = mu[2*l+1]
            std = np.nanstd(ob[:,:,0],axis=0)
            std[::2] = std[2*l]
            std[1::2] = std[2*l+1]
            ob[:,:,0] = (ob[:,:,0] - mu[np.newaxis]) / std[np.newaxis]
        if return_time:
            x = np.array([np.array(self.time[h:(m2+h)])[:,np.newaxis] for h in pred_horizon])
            time_targets = np.concatenate(x,axis=1)
            time_ob = np.array(self.time[0 if log_return_method is None else 1:-m1+1])[...,np.newaxis]
            return ob[:-m1], target, time_ob, time_targets
        return ob[:-m1], target, None, None
        
    
    def get_binned_statistic(self,bin_edges=None,x=None,statistic='sum', bid_negativ=False):
        """
        Computes simple binned statistic.
        
        For further information about the input variable, see documentation of 
        scipy.stats.binned_statistic 
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html)
        
        Parameters
        ----------
        bin_edges : numpy.array
            edges of the intervalls in which the prices are binnned.
        x : numpy.array, None
            A sequence of values to be binned in x[:,:,0] and data on which the 
            statistic will be computed in x[:,:,1]. If x is None self.ob is used.
        statistic : string or callable
            Possible statistics 'mean','median','count','sum','min','max', see 
            scipy.stats.binned_statistic.
            #TODO
            Default: 'sum'
        Returns
        -------
        stat: numpy.array
            Aggregated values of the binned values.
        """
        if bid_negativ:
            raise NotImplementedError('bid_negativ is not yet implemented, keep default False') #FIXME
        if x is None:
            x = np.array(self.ob)
        if False:
            x = self.ob
            print(x is self.ob)
            x = np.array(self.ob,copy=True)
            print(x is self.ob)
            mean = self.get_mean_price().mean()
            std = self.get_mean_price().std()
            lam = 25
            bin_edges = np.array([np.linspace(i-lam*std/5,i+lam*std/5,num = 2*lam + 1) for i in self.get_mean_price()])
            bin_edges = np.linspace(mean-lam*std/5,mean+lam*std/5,num = 2*lam + 1)
            x[:,1::2,1]*=-1
        #bin_edges = MaxNLocator(nbins=2*lam + 1).tick_values(mean-lam*std/5, mean+lam*std/5)
        
        if bin_edges.ndim == 1:
            stat = np.array([
                    binned_statistic(
                        x = tmp[:,0],
                        values = tmp[:,1],
                        statistic = statistic,
                        bins = bin_edges,
                        range = [bin_edges[0],bin_edges[-1]]
                    ).statistic for tmp in x
            ])
        elif bin_edges.ndim == 2:
            ext = [bin_edges.min(),bin_edges.max()]
            stat = np.array([
                    binned_statistic(
                        x = tmp[:,0],
                        values = tmp[:,1],
                        statistic = statistic,
                        bins = bin_edges[i],
                        range = ext
                    ).statistic for i,tmp in enumerate(x)
            ])
        else:
            raise ValueError('bin_edges in get_binned_statistic can only have 1 or 2 dimensions')
        return stat
        
if __name__ == '__main__':
    PATH_DATA = 'Data/Matlab'

    if False:    
        os.chdir('C:/Users/N/Documents/GitHub/ob_nw')
    
    
    file_path = os.path.join(PATH_DATA,sorted(os.listdir(PATH_DATA))[0])
    file_path

    a = LOB(file_path=file_path)
    b = a.get_data_targets(log_return_method='level_1_prices',return_time=True)
    print([i.shape for i in b])
    b = a.get_data_targets(log_return_method='same_level_prices',return_time=True)
    print([i.shape for i in b])
    b = a.get_data_targets(return_time=True)
    print([i.shape for i in b])
    self =a 
    
    
    a.get_centered_prices()
    a.check_date_correct()
    a.check_intraday_complete()    
    ret_ob, _ = a.get_returns()
    X,y = a.get_feature_y(ob=ret_ob,levels=1,lag=0)
    X.shape
    
    #TODO: https://stackoverflow.com/questions/27912801/how-to-pass-date-array-to-pcolor-plot
    
if False:
    #[a.gen['train'].lob.get_data_targets(**a.model_params['lob_model'])[0][:,:2,0].std() for d,i in enumerate(a.gen['train']) if d%183==0]
    s=[]
    for d,i in enumerate(a.gen['train']):
        if d%183==0:
            tmp = a.gen['train'].lob.get_data_targets(**a.model_params['lob_model'])[0][:,:2,0].flatten()
            s.append(tmp[tmp!=0].std())
            
    import matplotlib.pyplot as plt
    plt.plot([i*10**3 for i in s])