from typing import Tuple
from cachetools import TTLCache, cached
from cachetools.keys import hashkey 
import flask
import os
import urllib3
import json
import datetime
import pandas as pd
import numpy as np
import logging

API_URL = os.getenv('SOLAR_API_URL', 'http://raspberrypi.local:5555')
http = urllib3.PoolManager()

logger = logging.getLogger('solar-predictor')

def fetch_solar_data(dt: datetime.datetime) -> pd.DataFrame:
    dt = datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)
    response = http.request('GET', f"{API_URL}/solar/raw", fields={
        'datetime': dt,
    })

    data = json.loads(response.data)
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp')

def fetch_weather_data(dt: datetime.datetime) -> pd.DataFrame:
    dt = datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)
    response = http.request('GET', f"{API_URL}/weather/raw", fields={
        'datetime': dt,
    })

    data = json.loads(response.data)
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.set_index('timestamp')

def preprocess_dataset(solar_data: pd.DataFrame, weather_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    solar_data_hour = solar_data.resample('5min').mean()
    weather_data_hour = weather_data[['clouds']].resample('5min').mean()
    solar_data_hour['dt'] = solar_data_hour.index.to_series().diff().dt.total_seconds() / 3600 # in hours
    solar_data_hour['production_energy'] = solar_data_hour['webdata_now_p'] * solar_data_hour['dt'] # in Wh

    start = solar_data_hour.index.min()
    current_end = datetime.datetime.now()

    H_START = '04:00:00'
    H_END = '18:00:00'

    date = start.date()

    date_range = pd.date_range(start=f'{date} {H_START}', end=f'{date} {H_END}', freq='5min')

    date_range_df = pd.DataFrame(index=date_range)

    daily_data = date_range_df.join(solar_data_hour, how='left')
    daily_weather_data = date_range_df.join(weather_data_hour, how='left')
    daily_data = daily_data.join(daily_weather_data, how='left', rsuffix='_weather')
    daily_data = daily_data.ffill().bfill()

    assert daily_data.isna().sum().sum() == 0, "Data contains NaN values after forward and backward fill"
    assert len(daily_data) == 169, "Expected 169 rows of data"

    # find index that is closest to the current_end
    closest_index = daily_data.index.get_indexer([current_end], method='nearest')[0]

    return daily_data, date_range_df, int(closest_index)

def prep_sample(dataframe, index) -> Tuple[np.ndarray, np.ndarray, int]:
    dataframe = dataframe[['production_energy', 'clouds']].copy()
    dataframe['time'] = (dataframe.index.hour + dataframe.index.minute/60) / 24
    dataframe['month'] = dataframe.index.month/12
    dataframe['production_energy'] = dataframe['production_energy']/1000 # kWh
    dataframe['clouds'] = dataframe['clouds']/100
    dataframe = dataframe.ffill().bfill()

    mask = np.ones((169, len(dataframe.columns)), dtype=np.float32)
    mask[index:] = 0.0

    data = np.array(dataframe.values.astype(np.float32))
    data = data * mask
 
    return data, mask, index

def load_dataset(dt: datetime.datetime, current_overload=None) -> Tuple[np.ndarray, np.ndarray, int, pd.DataFrame]|None:
    try:
        solar = fetch_solar_data(dt)
    except Exception as e:
        logger.error(f"Error fetching solar data: {e}")
        return None
    
    try:
        weather = fetch_weather_data(dt)
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return None
    
    dataset, date_range_df, current = preprocess_dataset(solar, weather)

    if current_overload is not None:
        current = int(current_overload)
        

    d,m,i = prep_sample(dataset, current)

    return d, m, i, date_range_df

import torch as th
import torch.nn as nn

class AutoRegressiveTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_dim, dropout=0.1, seq_len=169):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_emb = nn.Parameter(th.randn(1, seq_len, model_dim))  
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, input_dim)  # output same dim as input for prediction

    def generate_square_subsequent_mask(self, sz):
        # Causal mask
        return th.triu(th.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, x):
        B, T, F = x.size()
        x = self.input_proj(x) + self.pos_emb[:, :T, :]
        causal_mask = self.generate_square_subsequent_mask(T).to(x.device)
        output = self.transformer(x, x, tgt_mask=causal_mask)
        # get last token (output)
        output = output[:, -1, :].unsqueeze(1)  # (B, 1, model_dim)
        return self.output_proj(output)


def load_model(path) -> AutoRegressiveTransformer:
    print(f"Loading model from {path}")
    model = AutoRegressiveTransformer(
        input_dim=4, 
        model_dim=16, 
        num_heads=8, 
        num_layers=3,
        ff_dim=96, 
        dropout=0.1,
        seq_len=169,
    )
    model.load_state_dict(th.load(path, weights_only=True, map_location=th.device('cpu')))
    return model


# model_normal = load_model('final-full.pth')
# model_optimistic = load_model('final-filtred.pth')
# model_pessimistic = load_model('final-filtred-pesimistic.pth')

model_normal = load_model(os.getenv('MODEL_NORMAL_PATH', 'final-full.pth'))
model_optimistic = load_model(os.getenv('MODEL_OPTIMISTIC_PATH', 'final-filtred.pth'))
model_pessimistic = load_model(os.getenv('MODEL_PESSIMISTIC_PATH', 'final-filtred-pesimistic.pth'))

logger.info("Models loaded")

def auto_regression(model, dataset: Tuple[np.ndarray, np.ndarray, int]):
    data, mask, mask_idx = dataset
    data = data.copy()

    with th.no_grad():
        dataset = th.from_numpy(data).to(th.float32).unsqueeze(0) # type: ignore[assignment]
        mask = th.from_numpy(mask).to(th.float32).unsqueeze(0)

        for _ in range(168 - mask_idx):
            output = model(dataset)
            dataset[:, mask_idx, :] = output.squeeze(1)
            mask[:, mask_idx, :] = 0
            mask_idx += 1

    dataset = dataset.numpy()[0]

    solar_data = dataset[:, 0] * 1000

    solar_data = np.clip(solar_data, 0, None)

    return solar_data 
        
def to_dataframe(dates, normal_pred, optimistic_pred, pessimistic_pred):
    df = pd.DataFrame({
        'normal': normal_pred,
        'optimistic': optimistic_pred,
        'pessimistic': pessimistic_pred,
        'timestamp': dates.to_series().dt.strftime('%Y-%m-%d %H:%M:%S')
    }, index=dates, columns=['normal', 'optimistic', 'pessimistic', 'timestamp'])

    df.index.name = 'timestamp'

    # convert to Watts
    df['dt'] = df.index.to_series().diff().dt.total_seconds() / 3600 #
    df['normal'] = df['normal'] / df['dt']
    df['optimistic'] = df['optimistic'] / df['dt']
    df['pessimistic'] = df['pessimistic'] / df['dt']

    return df

def calc_total_energy(series: pd.Series) -> float:
    return series.sum() / 1000  # convert to kWh

def get_acc(mask_idx):
    if mask_idx < 16:
        return "not_enough_data"
    if mask_idx < 48:
        return "not_accurate"
    
    return "optimal"

@cached(cache=TTLCache(maxsize=32, ttl=300), key=lambda _, dt, dataset, simple: hashkey((dt.year, dt.month, dt.day, simple)))
def run_autoregression_cached(model, dt, dataset: Tuple[np.ndarray, np.ndarray, int, pd.DataFrame], simple=False) -> dict:
    start_time = datetime.datetime.now()
    
    data, mask, mask_idx, time_range = dataset

    logger.info(f"Prediction for {dt} with mask_idx {mask_idx} in {'simple' if simple else 'full'} mode")

    data_normal = auto_regression(model_normal, (data, mask, mask_idx))
    data_optimistic = auto_regression(model_optimistic, (data, mask, mask_idx)) if model_optimistic is not None else None
    data_pessimistic = auto_regression(model_pessimistic, (data, mask, mask_idx)) if model_pessimistic is not None else None

    energy_normal = calc_total_energy(data_normal)
    energy_optimistic = calc_total_energy(data_optimistic) if data_optimistic is not None else None
    energy_pessimistic = calc_total_energy(data_pessimistic) if data_pessimistic is not None else None

    df = to_dataframe(
        dates=time_range.index,
        normal_pred=data_normal,
        optimistic_pred=data_optimistic if data_optimistic is not None else None,
        pessimistic_pred=data_pessimistic if data_pessimistic is not None else None,
    )

    end_time = datetime.datetime.now()

    return {
        "status": "success",
        "data": df.to_dict(orient='records'),
        "energy": {
            "normal": energy_normal,
            "optimistic": energy_optimistic,
            "pessimistic": energy_pessimistic,
        },
        "prediction_start": time_range.index[mask_idx].strftime('%Y-%m-%d %H:%M:%S'),
        "prediction_end": time_range.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
        "prediction_accuracy": get_acc(mask_idx),
        "prediction_index": mask_idx,
        "prediction_time": (end_time - start_time).total_seconds(),
        "mode": "simple" if simple else "full",
    }

def make_prediction(dt: datetime.datetime, prediction_start=None, simple=False) -> dict:
    dataset = load_dataset(dt, current_overload=prediction_start)

    if dataset is None:
        return {
            'status': 'error',
            'message': 'Failed to load dataset'
        }
    
    return run_autoregression_cached(model_normal, dt, dataset, simple=simple)

app = flask.Flask(__name__)

@app.route('/solar/predict', methods=['GET'])
def predict():
    dt = flask.request.args.get('datetime', default=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')

    prediction_start = flask.request.args.get('prediction_start', default=None, type=str)
    simple = flask.request.args.get('simple', default=False, type=bool)

    result = make_prediction(dt, prediction_start=prediction_start, simple=simple)
    return  json.dumps(result, default=str)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5554)