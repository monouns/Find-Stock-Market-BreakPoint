import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from gym import Env
from gym.spaces import Discrete, Box
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DNN import DNN, MyDataset

class StockEnv(Env):
    def __init__(self, df, scaler, result_model, result_img, transaction=False):
        #action we can take (hold, buy sell)  
        self.df = df.copy()
        self.transaction=transaction
        self.result_model = result_model
        self.result_img = result_img
        self.scaler_fe = scaler[0]
        self.scaler_close = scaler[1]
        
        self.df['return'] = 1.0
        self.df['acc_return'] = np.nan
    
        self.df['tf_signal'] = np.nan #for linear interpolation by rl action
        self.df.loc[0, 'tf_signal'] = self.df.loc[0, 'close']
        
        self.df['tf_trade'] = np.nan #for linear interpolation by actual trading
        self.df.loc[0, 'tf_trade'] = self.df.loc[0, 'close']
        
        self.df['rl_predict'] = np.nan
        self.df['actual_trade'] = np.nan
        self.df['position'] = np.nan
        
        self.buyhold = pd.DataFrame(self.df[['close']].values, columns = ['close'])
        self.buyhold['macd_acc'] = self.df['macd_acc']
        self.df.drop(columns=['macd_acc'], inplace=True)
        
        self.action_space = Discrete(3)
        
        # set state and label
        self.space = self.df.iloc[:,:-7]
        
        #current state 
        self.state = self.space.iloc[0].values       
        
        #observation range
        self.observation_space = Box(low = -np.inf, high = np.inf, shape =(1,1,15))    
        
        # set index(cnt), length
        self.index = 0
        self.length = len(self.df)-1
        self.done = False
        
        #trading
        self.buy =  0.0
        self.sell = 0.0
        self.rtn = 1.0
        self.reward = 1.0
        self.position = 0
        
        self.buyhold.loc[1:,'pct_change'] = self.buyhold[['close']].pct_change(1)[1:]['close']
        acc_return2 = 1.0
        self.buyhold.loc[0, 'acc_return'] = acc_return2
        for x in self.buyhold.index[1:]:
            if self.buyhold.loc[x, 'pct_change']+1:
                rtn = self.buyhold.loc[x, 'pct_change']+1
                acc_return2 = acc_return2 * rtn
                self.buyhold.loc[x, 'acc_return'] = acc_return2
            else:
                self.buyhold.loc[x, 'acc_return'] = acc_return2
        self.buyhold.loc[len(self.buyhold)-1, 'acc_return'] = acc_return2

    def step(self, action, predict = False):
        self.action = action

        if self.action == 0: #hold
            self.df.loc[self.index, 'rl_predict'] = self.action
            
            if self.buy != 0.0: #long position hold
                #print(self.index, "long hold")
                self.rtn = ((self.df.loc[self.index, 'close'] - self.df.loc[self.index-1, 'close']) / self.df.loc[self.index-1, 'close'] + 1)
                self.df.loc[self.index, 'return'] = self.rtn
                self.df.loc[self.index, 'actual_trade'] = 0

                self.reward *= self.rtn
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'position'] = 1
                
                
            elif self.sell != 0.0: #short position hold
                #print(self.index, "short hold")
                self.rtn =  ((self.df.loc[self.index-1, 'close'] - self.df.loc[self.index, 'close']) /self.df.loc[self.index, 'close']+ 1)
                self.df.loc[self.index, 'return'] = self.rtn
                self.df.loc[self.index, 'actual_trade'] = 0

                self.reward *= self.rtn
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'position'] = 2
                
                
            else: #no position
                #print(self.index, "no hold")
                #apply market fluctuation
                if self.index != 0:
                    self.rtn = ((self.df.loc[self.index, 'close'] - self.df.loc[self.index-1, 'close']) / self.df.loc[self.index-1, 'close'] + 1)
                else:
                    self.rtn = 1.0
                self.df.loc[self.index, 'return'] = self.rtn
                self.df.loc[self.index, 'actual_trade'] = 0
                
                self.reward *= self.rtn
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'position'] = 0
                
        #long position (buy first - sell next)
        elif self.action ==1: 
            self.df.loc[self.index, 'rl_predict']  = self.action 
            self.df.loc[self.index, 'tf_signal'] = self.df.loc[self.index, 'close']
            
            #when nothing hold, you have to buy first
            if self.sell==0.0 and self.buy == 0.0: 
                #print(self.index, "long start buy (buy)")
                self.buy=(self.df.loc[self.index, 'close'])

                self.rtn = 1.0 
                if self.transaction == True:
                    self.rtn -= self.rtn*0.002
                self.df.loc[self.index, 'return'] = self.rtn

                self.df.loc[self.index, 'actual_trade']  = 1
                self.df.loc[self.index, 'tf_trade'] = self.df.loc[self.index, 'close']  

                self.reward *= self.rtn
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'position'] = 1
                
            #when holding sth you have to buy finally
            elif self.sell != 0.0 and self.buy == 0.0:    
                pct = ((self.sell-self.df.loc[self.index, 'close']) /self.df.loc[self.index, 'close'] + 1) 
                self.rtn = ((self.df.loc[self.index-1, 'close'] - self.df.loc[self.index, 'close']) / self.df.loc[self.index, 'close'] + 1)
                
                self.buy = self.df.loc[self.index, 'close']
                    
                self.df.loc[self.index, 'return'] = self.rtn
                self.df.loc[self.index, 'actual_trade']  = 2
                self.df.loc[self.index, 'tf_trade'] = self.df.loc[self.index, 'close']  
                
                self.reward *= self.rtn
                if self.transaction==True:
                    self.reward -= pct*0.003 #for transaction cost
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'position'] = 2
                
                self.buy = 0.0
                
            else:  #long position hold
                #print(self.index, "long hold (buy)!")
                self.rtn = ((self.df.loc[self.index, 'close'] -self.df.loc[self.index-1, 'close']) / self.df.loc[self.index-1, 'close'] + 1)
                    
                self.df.loc[self.index, 'return'] = self.rtn
                self.df.loc[self.index, 'actual_trade'] = 0

                self.reward *= self.rtn
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'position'] = 1
                
            self.sell = 0.0
    
        #short position (sell first - buy next)        
        elif self.action ==2: 
            self.df.loc[self.index, 'rl_predict']  = self.action 
            self.df.loc[self.index, 'tf_signal'] = self.df.loc[self.index, 'close']
            
            #when holding sth you have to sell first
            if self.buy !=0.0 and self.sell==0.0: 
                #print(self.index, "long end sell(sell)!")
                pct = ((self.df.loc[self.index, 'close'] - self.buy) / self.buy + 1) 
                self.rtn = ((self.df.loc[self.index, 'close'] - self.df.loc[self.index-1, 'close']) / self.df.loc[self.index-1, 'close'] + 1)
                
                self.sell = self.df.loc[self.index, 'close']
                    
                self.df.loc[self.index, 'return'] = self.rtn
                self.df.loc[self.index, 'actual_trade']  = 2
                self.df.loc[self.index, 'tf_trade'] = self.df.loc[self.index, 'close']  
                
                self.reward *= self.rtn
                if self.transaction==True:
                    self.reward -= pct*0.003 #for transaction cost
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'position'] = 1
                
                self.sell = 0.0
            
            #when nothing hold, you have to sell first
            elif self.buy == 0.0 and self.sell == 0.0:
                self.sell = self.df.loc[self.index, 'close']
                self.rtn = 1.0 

                if self.transaction == True:
                    self.rtn -= self.rtn*0.002
                self.df.loc[self.index, 'return'] = self.rtn

                self.df.loc[self.index, 'actual_trade']  = 1
                self.df.loc[self.index, 'tf_trade'] = self.df.loc[self.index, 'close']  

                self.reward *= self.rtn
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'position'] = 2
            
            #when short holding, keep hold
            else: 
                #short position hold
                #print(self.index, "no hold (sell)")
                self.rtn = ((self.df.loc[self.index-1, 'close'] -self.df.loc[self.index, 'close']) / self.df.loc[self.index, 'close'] + 1)
                  
                self.df.loc[self.index, 'return'] = self.rtn
                self.df.loc[self.index, 'actual_trade'] = 0

                self.reward *= self.rtn
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'position'] = 2
        
            self.buy = 0.0
            
        self.state = self.space.iloc[self.index].values
        
        
        self.index += 1
        if self.index > self.length-1:  
            if self.buy : #long sell
                self.df.loc[self.index, 'rl_predict']  = self.action
        
                self.sell = self.df.loc[self.index, 'close']
                self.df.loc[self.index, 'actual_trade']  = 2
                
                pct = ((self.sell - self.buy) / self.buy + 1) 
                self.rtn = ((self.df.loc[self.index, 'close'] - self.df.loc[self.index-1, 'close']) / self.df.loc[self.index-1, 'close'] + 1)
                self.df.loc[self.index, 'return'] = self.rtn
                
                self.reward *= self.rtn
                if self.transaction == True:
                    self.reward -= pct * 0.003 #for transaction cost
                self.df.loc[self.index, 'acc_return'] = self.reward
                
                self.df.loc[self.index, 'tf_trade'] = self.df.loc[self.index, 'close']  
            
                self.buy = 0.0
                self.sell = 0.0
            
            self.df.loc[len(self.df)-1, 'tf_trade'] = self.df.loc[len(self.df)-1, 'close']
            self.df.loc[len(self.df)-1, 'tf_signal'] = self.df.loc[len(self.df)-1, 'close']
            
            self.df['acc_return'] = self.df['acc_return'].interpolate(method='linear', limit_direction='both')
            acc_return = self.df.loc[len(self.df)-1, 'acc_return']
            print("Cumulative Return: ", acc_return)    
            
            #regression model reward
            loss2, self.df.loc[:, 'tf_signal'], self.df.loc[:, 'tf_trade'] = self.loss(predict)
            self.reward -= loss2*100
            
            if predict:
                self.render()
                self.render1()
                self.render2()
                self.render3()
            
            self.done = True
        else:
            self.done = False

        info = {} 

        return self.state, self.reward, self.done, info
    
    def loss(self, predict = False):
        regression_ = self.df.copy()
        
        #interpolation code
        regression_['tf_signal'] = regression_['tf_signal'].interpolate(method='linear', limit_direction='both')
        regression_['tf_trade'] = regression_['tf_trade'].interpolate(method='linear', limit_direction='both')

        tf_signal = regression_['tf_signal'].copy()
        tf_trade = regression_['tf_trade'].copy()
        
        regression_fe = regression_.iloc[:, :-3].copy()
        regression_close = regression_.loc[: ,['close']].copy()
        
        if predict==True:
            regression = pd.DataFrame(self.scaler_fe.transform(regression_fe.values), columns=regression_fe.columns)
            regression['close'] = pd.DataFrame(self.scaler_close.transform(regression_close.values))
            regression['tf_signal'] = pd.DataFrame(self.scaler_close.transform(regression_.loc[: ,['tf_signal']].copy().values))
            regression['tf_trade'] = pd.DataFrame(self.scaler_close.transform(regression_.loc[: ,['tf_trade']].copy().values))
        else:
            self.scaler_close = MinMaxScaler()
            self.scaler_fe = MinMaxScaler()
            
            self.scaler_close.fit(regression_close.values)
            self.scaler_fe.fit(regression_fe.values)

            regression = pd.DataFrame(self.scaler_fe.transform(regression_fe.values), columns=regression_fe.columns)
            regression['close'] = pd.DataFrame(self.scaler_close.transform(regression_close.values))
            regression['tf_signal'] = pd.DataFrame(self.scaler_close.transform(regression_.loc[: ,['tf_signal']].copy().values))
            regression['tf_trade'] = pd.DataFrame(self.scaler_close.transform(regression_.loc[: ,['tf_trade']].copy().values))
        
        regression[regression_.columns[-3]] = regression_.iloc[:, -3].copy()
        regression[regression_.columns[-2]] = regression_.iloc[:, -2].copy()
        regression[regression_.columns[-1]] = regression_.iloc[:, -1].copy()

        plt.figure(figsize=[20,10])
        plt.plot(regression.index, regression['close'])
        plt.plot(regression.index, regression['tf_trade'], alpha=0.3, color = 'purple', linewidth=5.0 , label = 'linear_tf')
        plt.scatter(regression[regression['actual_trade']!=0].index, regression[regression['actual_trade']!=0]['close'], marker='o', color='red', s=100, label = 'break point')
        #plt.scatter(regression[regression['rl_predict']==2].index, regression[regression['rl_predict']==2]['close'], marker='v', color='red', s=100, label = 'sell_signal')
        plt.legend(fontsize=15)
        plt.xticks(fontsize=20, rotation=45)
        plt.xlabel("Time", fontsize=35)
        plt.ylabel("Close", fontsize=35)
        plt.yticks(fontsize=20, rotation=45)
        plt.title("Linear Trend Filtering", fontsize=45)
        plt.savefig(os.path.join(self.result_img, f'Local Linear TF-wtrading.png'))
        plt.show()
        
        regression = regression.iloc[1:-1]
        
        #get regression loss
        print("Regression model training...")        
        num_points_for_train = 60 # look back
        offset = 5
        target_col = regression.columns.get_loc("close") # target column index
        batch_size = 2056
        num_workers = 4
        pin_memory = True
        num_epoch = 100
        lr = 1e-4
        
        train_dataset = MyDataset(regression.values, 
                                  num_points_for_train, offset, target_col)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             drop_last=False, 
                                             num_workers=num_workers, pin_memory=pin_memory)
        
        model = DNN(in_features= 21,
            hidden_dim = 64,
            out_features=1).cuda()
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        if predict == True:
            checkpoint = torch.load(os.path.join(self.result_model, f'prediction-best.pt'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            criterion = checkpoint['loss']
            model = model.eval()
            with torch.no_grad():
                test_loss1 = []
                test_loss2 = []
                test_loss3 = []
                test_loss4 = []

                answer = []
                prediction = []

                for i, (x,y) in enumerate(train_loader):
                    x = x.cuda()
                    y = y.cuda().reshape(-1,1)
                    answer.extend(y.detach().cpu().numpy())

                    outputs = model(x)
                    prediction.extend(outputs.reshape(-1,1).detach().cpu().numpy())
                    loss = criterion(outputs, y)

                    criterion3 = nn.L1Loss()
                    loss3 = criterion3(input=outputs, target = y)

                    loss4 = torch.mean(torch.abs(torch.subtract(outputs, y) / y))

                    test_loss1.append(loss.item())
                    test_loss2.append(torch.sqrt(loss).item())
                    test_loss3.append(loss3.item())
                    test_loss4.append(loss4.item())

                answer = self.scaler_close.inverse_transform(answer)
                prediction = self.scaler_close.inverse_transform(prediction)
                
                plt.figure(figsize=[20,10])
                plt.plot(answer, linestyle=':', color = 'black', alpha = 0.5, label = 'truth')
                plt.plot(prediction, color = 'blue', label = 'prediction')
                plt.legend()
                plt.title("Regression with RL", fontsize=45)
                plt.show()

                answer = torch.Tensor(answer)
                prediction = torch.Tensor(prediction)

                loss = criterion(answer, prediction)
                criterion3 = nn.L1Loss()
                loss3 = criterion3(input=prediction, target = answer)

                loss4 = torch.mean(torch.abs(torch.subtract(prediction, answer) / answer))
                final_loss = np.mean(test_loss1)

                print("MSE: ", loss.item())
                print("rMSE: ", torch.sqrt(loss).item())
                print("MAE: ", loss3.item())
                print("MAPE: ", loss4.item())
                
        else:    
            for epoch in tqdm(range(0, num_epoch)):
                train_loss = []
                answer = []
                prediction = []
                model = model.train()
                for i, (x,y) in enumerate(train_loader):
                    x = x.cuda()
                    y = y.cuda().reshape(-1,1)
                    answer.extend(y.detach().cpu().numpy())

                    outputs = model(x)
                    prediction.extend(outputs.reshape(-1,1).detach().cpu().numpy())
                    loss = torch.sqrt(criterion(outputs, y))

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    train_loss.append(loss.item())
            
            answer = self.scaler_close.inverse_transform(answer)
            prediction = self.scaler_close.inverse_transform(prediction)
            
            plt.figure(figsize=[20,10])
            plt.plot(answer, linestyle=':', color = 'black', alpha = 0.5, label = 'truth')
            plt.plot(prediction, color = 'blue', label = 'prediction')
            plt.legend()
            
            plt.title("Regression with RL", fontsize=45)
            plt.show()
            
            final_loss = np.mean(train_loss)
            print(final_loss)
            
            torch.save({
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optim.state_dict(),
                'loss' : criterion
                }, os.path.join(self.result_model, f'prediction-best.pt'))
            
        return final_loss, tf_signal, tf_trade


    def render(self):
        plt.figure(figsize=[20,10])
        plt.plot(self.df.index, self.df['close'])
        #plt.plot(self.df.index, self.df['tf'], alpha=0.3, color = 'purple', linewidth=5.0 , label = 'linear_tf')
        plt.scatter(self.df[self.df['actual_trade']==1].index, self.df[self.df['actual_trade']==1]['close'], marker='^', color='green', s=200, label = 'buy')
        plt.scatter(self.df[self.df['actual_trade']==2].index, self.df[self.df['actual_trade']==2]['close'], marker='v', color='red', s=200, label = 'sell')
        plt.legend(fontsize=15)
        plt.xticks(fontsize=20, rotation=45)
        plt.xlabel("Time", fontsize=35)
        plt.yticks(fontsize=20, rotation=45)
        plt.ylabel("Close", fontsize=35)
        plt.title("Actual Trading", fontsize=45)
        plt.savefig(os.path.join(self.result_img,f"Actual Trading.png"))
        plt.show()
        
    def render1(self):
        plt.figure(figsize=[20,10])
        plt.plot(self.df.index, self.df['close'])
        #plt.plot(self.df.index, self.df['tf'], alpha=0.3, color = 'purple', linewidth=5.0 , label = 'linear_tf')
        plt.scatter(self.df[self.df['rl_predict']==1].index, self.df[self.df['rl_predict']==1]['close'], marker='^', color='green', s=100, label = 'buy_signal')
        plt.scatter(self.df[self.df['rl_predict']==2].index, self.df[self.df['rl_predict']==2]['close'], marker='v', color='red', s=100, label = 'sell_signal')
        plt.legend(fontsize=15)
        plt.xticks(fontsize=20, rotation=45)
        plt.xlabel("Time", fontsize=35)
        plt.yticks(fontsize=20, rotation=45)
        plt.ylabel("Close", fontsize=35)
        plt.title("Trading Signal", fontsize=45)
        plt.savefig(os.path.join(self.result_img, f'Trading Signal.png'))
        plt.show()
        
    def render2(self):
        #baseline: buy-hold  
        plt.figure(figsize= [20,10])
        plt.plot(self.buyhold.index, self.buyhold['acc_return'], linestyle=':', c = 'black', label = 'buy-hold')
        #plt.plot(self.buyhold.index, self.buyhold['macd_acc'],linestyle=':', c = 'grey', label = 'MACDwNoCost')
        plt.plot(self.df.index[1:], self.df['acc_return'][1:], label = 'RL')
        plt.legend(fontsize=15)
        plt.xticks(fontsize=20, rotation=45)
        plt.xlabel("Time", fontsize=35)
        plt.yticks(fontsize=20, rotation=45)
        plt.ylabel("Cumulative Return", fontsize=35)
        plt.title("Cumulative Return", fontsize=45)
        plt.savefig(os.path.join(self.result_img, f'Cumulative Return.png'))
        plt.show()
        
        print("BuyHold cumulative return: ", self.buyhold['acc_return'][len(self.buyhold)-1])
        print("RL cumulative return: ", self.df['acc_return'][len(self.df)-1])
        
    def render3(self):
        plt.figure(figsize=[20,10])
        plt.plot(self.df.index, self.df['close'])
        #plt.plot(self.df.index, self.df['tf'], alpha=0.3, color = 'purple', linewidth=5.0 , label = 'linear_tf')
        plt.scatter(self.df[self.df['position']==1].index, self.df[self.df['position']==1]['close'], marker='^', color='green', s=100, label = 'long position')
        plt.scatter(self.df[self.df['position']==2].index, self.df[self.df['position']==2]['close'], marker='v', color='red', s=100, label = 'short position')
        plt.legend(fontsize=15)
        plt.xticks(fontsize=20, rotation=45)
        plt.xlabel("Time", fontsize=35)
        plt.yticks(fontsize=20, rotation=45)
        plt.ylabel("Close", fontsize=35)
        plt.title("Long/Short Position", fontsize=45)
        plt.savefig(os.path.join(self.result_img, f'Long-Short Position.png'))
        plt.show()
    
    def get_data(self):
        return self.df, [self.scaler_fe, self.scaler_close]

    def reset(self):        
        self.df['rl_predict'] = np.nan
        self.df['actual_trade'] = np.nan
        
        self.df['return'] = 1.0
        self.df['acc_return'] = np.nan
        
        self.df['tf_trade'] = np.nan
        self.df.loc[0, 'tf_trade'] = self.df.loc[0, 'close']
        self.df['tf_signal'] = np.nan
        self.df.loc[0, 'tf_signal'] = self.df.loc[0, 'close']
        
        self.df['position'] = np.nan
        
        # set start state
        self.space = self.df.iloc[:,:-7]
        
         #current state and label
        self.state = self.space.iloc[0].values  
        
        # set index(cnt), length
        self.index = 0
        self.length = len(self.df)-1
        self.done = False
        
        #trading
        self.buy =0.0
        self.sell = 0.0
        self.rtn = 1.0
        self.reward = 1.0
        self.position=0

        return self.state