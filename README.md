# LOBDeepPP

This repository contains some core codes used in the creation of my master thesis with the topic 'Forecasting Stock Prices of Limit Order Book Data with Deep Neural Networks'. I can not provide the used LOB data since I am not allowed to share these. 

The main program is `LOBDeepPP_train.py` which is a CLI supporting script, the arguments are the model number and a parameter file, see
```bash
python LOBDeepPP_train.py --help
```
e.g. 
```
python LOBDeepPP_train.py --model 17a --path_params LOBDeepPP/LOBDeepPP_params_files/params_L10a.json
```
