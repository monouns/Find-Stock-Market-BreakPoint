import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import sys 
sys.path.append("./l1tf")
import pandas_wrapper

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from FeatureEngineering import FeatureEngineering
from StockEnv import StockEnv
from DNN import DNN, MyDataset

from stable_baselines3 import PPO, DQN, A2C

SEED=10
num_workers = 4
pin_memory = True
device = 'cuda'
data_folder = './data'

if __name__=='__main__':
    random_seed=10
    torch.manual_seed(random_seed) # for torch.~~
    torch.backends.cudnn.deterministic = True # for deep learning CUDA library
    torch.backends.cudnn.benchmark = False # for deep learning CUDA library
    np.random.seed(random_seed) # for numpy-based backend, scikit-learn
    random.seed(random_seed) # for python random library-based e.g., torchvision
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    
    # Arguments parsing
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpuidx', default=1, type=int, help='gpu index')
    parser.add_argument('--data', default='NASDAQ100', type=str, help='data name')
    parser.add_argument('--model', default='PPO', type=str, help='DQN | PPO | A2C')
    parser.add_argument('--transaction', default=0, type=int, help='transaction cost')
    parser.add_argument('--epoch', default=5, type=int, help='training epoch')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)
    
    root_folder = f'./check/{args.model}'
    result_model = os.path.join(root_folder, 'model')
    result_img = os.path.join(root_folder, 'img')
    os.makedirs(root_folder, exist_ok=True)
    os.makedirs(result_model, exist_ok=True)
    os.makedirs(result_img, exist_ok=True)
    os.makedirs(os.path.join(result_img, 'train'), exist_ok=True)
    os.makedirs(os.path.join(result_img, 'test'), exist_ok=True)
    
    fe = FeatureEngineering(os.path.join(data_folder, args.data+'_m.csv'))
    train, test = fe.get_data()
    
    epoch = args.epoch
    scaler_close = MinMaxScaler()
    scaler_fe = MinMaxScaler()
    
    env = StockEnv(train,[scaler_fe, scaler_close], result_model, result_img, args.transaction)
    
    if args.model == 'PPO':
        RLmodel = PPO("MlpPolicy", env, verbose=1, seed = SEED, device='cuda') 
    elif args.model == 'DQN':
        RLmodel = A2C("MlpPolicy", env, verbose=1, seed = SEED, device='cuda')
    elif args.model =='A2C':
        RLmodel = DQN("MlpPolicy", env, verbose=1, seed = SEED, device='cuda')
    
    RLmodel.learn(total_timesteps=len(train)*epoch)
    RLmodel.save(os.path.join(result_model,f"{args.model}_{args.epoch}"))
    
    _, scaler = env.get_data()
    
    ### train dataset prediction ###
    env1 = StockEnv(train, scaler, result_model, os.path.join(result_img,'train'), args.transaction)
    obs = env1.reset().reshape(1,1,15)
    while range(1):
        action, _states = RLmodel.predict(obs)
        obs, reward, done, info = env1.step(action, predict = True)
        obs = obs.reshape(1,1,15)
        if done:
            tmp1 = env1.get_data() 
            break
    
    ### test dataset prediction
    env2 = StockEnv(test, scaler, result_model, os.path.join(result_img, 'test'), args.transaction)
    obs = env2.reset().reshape(1,1,15)
    while range(1):
        action, _states = RLmodel.predict(obs)
        obs, reward, done, info = env2.step(action, predict=True)
        obs = obs.reshape(1,1,15)
        if done:
            tmp1 = env2.get_data() 
            break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    