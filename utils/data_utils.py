import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np
from abc import ABC

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

class DataLoader(Dataset, ABC):
    def __init__(self, batch_size, sample_len):
        self.batch_size = batch_size
        self.sample_len = sample_len
        hour_data = np.vstack([get_hour_data(2011, 1, 2013, 12), get_hour_data(2014, 7, 2017, 6)])
        weather_data = np.vstack([get_weather_data(2011, 1, 2013, 12), get_weather_data(2014, 7, 2017, 6)])
        scaler = StandardScaler()
        hour_data = scaler.fit_transform(hour_data)
        weather_data = scaler.fit_transform(weather_data)
        self.hour_data = hour_data
        self.weather_data = weather_data

    def process_data(self,data):
        samples = torch.zeros((1, self.sample_len, data.shape[1]))
        for i in range(len(data)):
            sample = data[i:i + self.sample_len]
            samples = torch.cat((samples, sample.unsqueeze(0)), dim=0)
        return samples[1:, :, :]

    def __getitem__(self, item):
        start = item * self.batch_size
        end = (item + 1) * self.batch_size
        if end > len(self.hour_data):
            end = len(self.hour_data)
        batch_data = self.hour_data[start:end]
        batch_weather = self.weather_data[start:end]
        return batch_data, batch_weather

    def __len__(self):
        return (len(self.hour_data) + self.batch_size - 1) // self.batch_size
    
def get_train_data(batch, sample_len, hour_data, weather_data):
    indices = np.arange(0,hour_data.shape[0] - sample_len + 1,sample_len)
    # indices = np.arange(hour_data.shape[0] - sample_len + 1)
    np.random.shuffle(indices)
    nodes_samples = torch.zeros((1, sample_len, hour_data.shape[1]))
    aux_samples = torch.zeros((1, sample_len, weather_data.shape[1]))
    count = 0
    index = 0
    for i in indices:
        node_sample = torch.from_numpy(hour_data[i:i + sample_len]).float()
        nodes_samples = torch.cat((nodes_samples, node_sample.unsqueeze(0)), dim=0)
        aux_sample = torch.from_numpy(weather_data[i:i + sample_len]).float()
        aux_samples = torch.cat((aux_samples, aux_sample.unsqueeze(0)), dim=0)
        count += 1
        index += 1
        if count == batch:
            print("第{}个batch".format(index // batch))
            yield nodes_samples[1:, :, :], aux_samples[1:, :, :]
            nodes_samples = torch.zeros((1, sample_len, hour_data.shape[1]))
            aux_samples = torch.zeros((1, sample_len, weather_data.shape[1]))
            count = 0

if __name__ == '__main__':
    # hour_data = np.vstack([get_hour_data(2011,1,2013,12),get_hour_data(2014,7,2017,6)])
    # print(get_weather_data(2011,1,2013,12).shape)
    batch_size = 4 
    hour_data = np.vstack([get_hour_data(2011, 1, 2013, 12), get_hour_data(2014, 7, 2017, 6)])
    weather_data = np.vstack([get_weather_data(2011, 1, 2013, 12), get_weather_data(2014, 7, 2017, 6)])
    length = hour_data.shape[1]
    scaler = StandardScaler()
    sca_hour_data = scaler.transform(hour_data)
    sca_weather_data = scaler.transform(weather_data)
    # for hour_data, weather_data in data_loader:
    for hour_data, weather_data in get_train_data(batch_size, length, sca_hour_data, sca_weather_data):
        train_data = hour_data[:, :length * 0.8, :]
        test_data = hour_data[:, length * 0.8:, :]

    print(hour_data.shape)
    print(weather_data.shape)
    print(train_data.shape)
    print(test_data.shape)