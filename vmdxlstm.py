import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from plotly import graph_objects as go

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        scale = 1.0 / np.sqrt(hidden_size)
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        concat = np.vstack((h_prev, x))
        ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        cct = np.tanh(np.dot(self.Wc, concat) + self.bc)
        ct = ft * c_prev + it * cct
        ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        ht = ot * np.tanh(ct)
        return ht, ct

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

class BiLSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forward_lstm = LSTMCell(input_size, hidden_size)
        self.backward_lstm = LSTMCell(input_size, hidden_size)
        self.Wy = np.random.randn(1, 2 * hidden_size) * 0.01
        self.by = np.zeros((1, 1))
        self.scaler = StandardScaler()

    def forward(self, X):
        T = len(X)
        hf_prev = np.zeros((self.hidden_size, 1))
        cf_prev = np.zeros((self.hidden_size, 1))
        hb_prev = np.zeros((self.hidden_size, 1))
        cb_prev = np.zeros((self.hidden_size, 1))

        h_forward = []
        for t in range(T):
            x = X[t].reshape(-1, 1)
            hf_prev, cf_prev = self.forward_lstm.forward(x, hf_prev, cf_prev)
            h_forward.append(hf_prev)

        h_backward = []
        for t in reversed(range(T)):
            x = X[t].reshape(-1, 1)
            hb_prev, cb_prev = self.backward_lstm.forward(x, hb_prev, cb_prev)
            h_backward.insert(0, hb_prev)

        h_combined = [np.vstack((hf, hb)) for hf, hb in zip(h_forward, h_backward)]
        y_pred = np.dot(self.Wy, h_combined[-1]) + self.by
        return y_pred

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        model_data = data['model']
        scaler_data = data['scaler']
        model = BiLSTM(model_data['input_size'], model_data['hidden_size'])

        model.forward_lstm.__dict__.update(model_data['forward_lstm'])
        model.backward_lstm.__dict__.update(model_data['backward_lstm'])
        model.Wy = model_data['Wy']
        model.by = model_data['by']
        model.scaler.mean_ = scaler_data['scaler_mean']
        model.scaler.scale_ = scaler_data['scaler_scale']
        return model

def prepare_data(data, window_size=7):
    X = [data[i:i+window_size] for i in range(len(data) - window_size)]
    y = [data[i+window_size] for i in range(len(data) - window_size)]
    return np.array(X), np.array(y)

def predict(model, X_test):
    predictions = []
    for x in X_test:
        y_pred = model.forward(x)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]  # Jika BiLSTM.forward() mengembalikan tuple
        y_pred = np.squeeze(np.array(y_pred))  # Pastikan menjadi float
        predictions.append(y_pred)
    return np.array(predictions)

def Hasil_prediksi_lstm_plot(path):
    df = pd.read_csv(path)
    model = BiLSTM.load_model("manual_bilstm_btc_model.h5")

    btc_prices = df['BTC_Close_BTC'].dropna().values.reshape(-1, 1)
    scaled_prices = model.scaler.transform(btc_prices).flatten()

    window_size = 7
    X, y = prepare_data(scaled_prices, window_size)
    split = int(0.8 * len(X))
    X_test = X[split:]
    y_test = y[split:]

    preds_scaled = predict(model, X_test)
    preds = model.scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_actual = model.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    test_dates = df['Date'][split + window_size:split + window_size + len(y_actual)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=y_actual, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=preds, mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
    fig.update_layout(
        title='Prediksi BTC (LSTM)',
        xaxis_title='Tanggal',
        yaxis_title='Harga BTC (USD)',
        template='plotly_dark'
    )
    return fig
