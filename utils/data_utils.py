import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):      
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

def dateToStr(date: int):
    if date < 10:
        return '0' + str(date)
    else:
        return str(date)

def get_hour_data(start_year: int, start_month: int, end_year: int, end_month: int):  # threshold
    matrix = None
    while start_year < end_year or \
            (start_year == end_year and start_month <= end_month):
        yearStr = str(start_year)
        monthStr = dateToStr(start_month)
        data = pd.read_csv(
            'GSA-forecastor\datasets\hour_data_matrix_' + yearStr + '/hour_data_matrix' + yearStr + '-' + monthStr + '.csv', sep=',',
            encoding="utf-8")
        matrix = data.values[:, 4:] if matrix is None else \
            np.concatenate((matrix, data.values[:, 4:]), axis=0)
        if (start_month + 1) % 12 == 1:
            start_month = 1
            start_year += 1
        else:
            start_month += 1
    return matrix


def get_weather_data(start_year: int, start_month: int, end_year: int, end_month: int):
    weather_data = pd.read_csv(
        'GSA-forecastor\datasets\\all_padding_ready_weather.csv', sep=',',
        encoding="utf-8")
    filter_data = weather_data.loc[
        (weather_data['year'] * 12 + weather_data['month'] >= start_year * 12 + start_month)
        & (weather_data['year'] * 12 + weather_data['month'] <= end_year * 12 + end_month)]
    matrix = filter_data.values[:, 4:]
    return matrix   

# hour_data = np.vstack([get_hour_data(2011, 1, 2013, 12), get_hour_data(2014, 7, 2017, 6)])
# weather_data = np.vstack([get_weather_data(2011, 1, 2013, 12), get_weather_data(2014, 7, 2017, 6)])
# houe_data: (52608, 67) weather_data: (52608, 5) all_data: (52608, 72)
# 划分数据集及标准化
def read_data(all_data, seq_len, scale = True):
    
    df = pd.DataFrame(all_data)
    scaler = None

    n_train = int(len(df[0]) * 0.8)
    n_test = int(len(df[0]) * 0.2)

    train_begin = 0 
    train_end = n_train

    test_begin = len(df) - n_test - seq_len
    test_end = len(df)

    if scale: 
        scaler = StandardScaler()
        train_data = df[0:n_train]
        scaler.fit(train_data.values)
        data = scaler.transform(df.values)
    else:
        data = df.values

    return data[train_begin:train_end], data[test_begin:test_end], scaler, [train_begin, test_begin]


# all_data: (52608, 72)
# 划分X与y
# 两种预测任务：
# 1、以 7 * 24 = 168h 预测接下来3h
# 2、以前四周同一天的数据 4 * 24 = 96h 预测接下来的3h
class SeqData(Dataset):
    def __init__(self, data, start, seq_len = 168, horizon = 3, flag = "hour"):
        self.data = data
        self.seq_len = seq_len 
        self.horizon = horizon
        self.start = start   
        self.flag = flag

    def __getitem__(self, index):
        seq_begin = index 
        seq_end = index + self.seq_len
        label_end = seq_end + self.horizon

        if self.flag == "hour":
            label_begin = seq_end + self.horizon - 3
        else:
            label_begin = label_end

        return self.data[seq_begin:seq_end], self.data[label_begin: label_end]

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

# 批处理
def get_dataloaders(train, test, starts, batch_size = 16, seq_len = 168, horizon = 3, flag = "hour"):

    train_data = SeqData(train, starts[0], seq_len, horizon, flag)
    test_data = SeqData(test, starts[1], seq_len, horizon, flag)
    
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, drop_last = True)
    # test_loader_one = DataLoader(test_data, batch_size = 1, shuffle = False, drop_last = False)

    # return train_loader, test_loader, test_loader_one
    return train_loader, test_loader

def final_get_dataloaders(start_year: int, start_month: int, end_year: int, end_month: int):
    hour_data = get_hour_data(start_year, start_month, end_year, end_month)
    weather_data = get_weather_data(start_year, start_month, end_year, end_month)

    hour_train_data, hour_test_data, scale, starts = read_data(hour_data, seq_len = 168)
    weather_train_data, weather_test_data, _, _= read_data(weather_data, seq_len = 168)

    hour_train_loader, hour_test_loader = get_dataloaders(hour_train_data, hour_test_data, starts, batch_size = 16, seq_len = 168, horizon = 3, flag = "hour")
    weather_train_loader, weather_test_loader = get_dataloaders(weather_train_data, weather_test_data, starts, batch_size = 16, seq_len = 168, horizon = 3, flag = "weather")

    return hour_train_loader, weather_train_loader, hour_test_loader, weather_test_loader, scale

if __name__ == '__main__':
    # hour_data = np.vstack([get_hour_data(2011,1,2013,12),get_hour_data(2014,7,2017,6)])
    # print(get_weather_data(2011,1,2013,12).shape)
    # batch_size = 3
    # hour_data = np.vstack([get_hour_data(2011, 1, 2013, 12), get_hour_data(2014, 7, 2017, 6)])
    # weather_data = np.vstack([get_weather_data(2011, 1, 2013, 12), get_weather_data(2014, 7, 2017, 6)])
    # sample_len = 24*7+3
    # scaler = StandardScaler()
    # sca_hour_data = scaler.transform(hour_data)
    # sca_weather_data = scaler.transform(weather_data)
    # print(hour_data.shape)
    # print(weather_data.shape)
    hour_train_loader, weather_train_loader, hour_test_loader, weather_test_loader, scale = final_get_dataloaders(2011,1,2019,12)
    print("=================================")
    for x, y in hour_train_loader:
        print(x.shape)
        print(y.shape)
        break
