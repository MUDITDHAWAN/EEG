import os
import pandas as pd
import numpy as np

def read_mat_files():
    os.chdir("/home/shreya/ArithmeticTask/ArithmeticTask_Data_set_csv")
    list_df = np.array([[[[]]]])
    list_y = []
    for data_file in sorted(os.listdir()):
        if("y" in data_file):
            y = pd.DataFrame(pd.read_csv(data_file))
            y.drop("Unnamed: 0", axis=1, inplace=True)
            list_y = y.to_numpy()
        else:
            df = pd.DataFrame(pd.read_csv(data_file))
            df.drop("Unnamed: 0", axis=1, inplace=True)
            if("00_1" in data_file):
                list_df = np.array([[df.to_numpy()]])
            else:
                list_df = np.append(list_df, np.array([[df.to_numpy()]]), axis = 0)
    return list_df, list_y


list_df, list_y = read_mat_files()

nlist_df = list_df.reshape(list_df.shape[0],list_df.shape[2], list_df.shape[3])

print(nlist_df.shape)

def convert_into_windows(window_size):
    Ytrain = np.zeros((1,1))
    Xtrain = np.zeros((1,window_size, 20))
    
    for i in range(0, nlist_df.shape[0]):
        for j in range(0, nlist_df.shape[2]//window_size):
            window = nlist_df[i,:,j*window_size: (j+1)*window_size]
            window = window.transpose()
            window = window.reshape(1, window.shape[0], window.shape[1])
            Xtrain = np.concatenate((Xtrain, window), axis = 0)
            
            if(i%2==1):
                Ytrain = np.concatenate((Ytrain,[[1.]] ), axis = 0)
            else:
                Ytrain = np.concatenate((Ytrain,[[0.]] ), axis = 0)
            
    Xnew = np.delete(Xtrain, (0), axis=0)
    Ynew = np.delete(Ytrain, (0), axis=0)
    
    print(Xtrain.shape)
    print(Ytrain.shape)
    
    return Xnew, Ynew


def save_window_data(Xtrain, Ytrain, window):
    os.chdir("/home/shreya/ArithmeticTask/LSTMmodel/Data/")
    file_name_X = "Xdata_" + str(window) + ".npy"
    file_name_Y = "Ydata_" + str(window) + ".npy"
    np.save(file_name_X, Xtrain)
    np.save(file_name_Y, Ytrain)


for window_size in [100, 200, 250, 500, 750, 1000]:
    Xtrain, Ytrain = convert_into_windows(window_size)
    
    save_window_data(Xtrain, Ytrain, window_size)