import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

def hasil_prediksi(path):
    # 1. Load dataset
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 2. Define features and target
    features = [
        'BTC_High_BTC', 'BTC_Low_BTC', 'BTC_Open_BTC',
        'GSPC_High_GSPC', 'GSPC_Close_GSPC', 'GSPC_Open_GSPC',
        'GSPC_Low_GSPC', 'XAU_Low_XAU', 'XAU_Open_XAU', 'XAU_Close_XAU',
        'XAU_High_XAU', 'XAU_Volume_XAU'
    ]
    target = 'BTC_Close_BTC'

    X_raw = df[features].values
    y_raw = df[target].values
    dates = df['Date']

    # 3. Split data
    split_idx = int(len(df) * 0.8)
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    # 4. Standardize
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train_raw)
    X_test = X_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    # 5. Initialize parameters
    def initialize_weights(n_features, seed=42):
        np.random.seed(seed)
        weights = np.random.randn(n_features) * 0.01
        bias = np.random.randn() * 0.01
        return weights, bias

    # 6. Predict batch
    def predict_for_trainntest(X, weights, bias):
        return np.dot(X, weights) + bias

    # 7. Compute gradients
    def compute_gradient(X_batch, y_batch, y_pred, weights, C, epsilon):
        n_samples = X_batch.shape[0]
        error = y_batch - y_pred
        grad_loss_w = weights * 0
        grad_loss_b = 0
        loss_indices = np.abs(error) > epsilon
        if np.any(loss_indices):
            sign_error = -np.sign(error[loss_indices])
            grad_loss_w = np.dot(X_batch[loss_indices].T, sign_error)
            grad_loss_b = np.sum(sign_error)
        grad_reg_w = weights
        delta_w = grad_reg_w + C * grad_loss_w / n_samples
        delta_b = C * grad_loss_b / n_samples
        return delta_w, delta_b

    # 8. Training function
    def train_svr(X_train, y_train, C=10, epsilon=0.1, learning_rate=0.0005, epochs=200, batch_size=64, seed=42):
        n_samples, n_features = X_train.shape
        weights, bias = initialize_weights(n_features, seed)
        print(f"Initial weights: {weights}")
        print(f"Initial bias: {bias}")
        
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                y_pred = predict_for_trainntest(X_batch, weights, bias)
                delta_w, delta_b = compute_gradient(X_batch, y_batch, y_pred, weights, C, epsilon)
                weights -= learning_rate * delta_w
                bias -= learning_rate * delta_b
        return weights, bias

    # 9. Train model
    weights, bias = train_svr(X_train, y_train)

    # 10. Predict
    y_pred_test_scaled = predict_for_trainntest(X_test, weights, bias)
    y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()

    # 11. Evaluation
    mae = mean_absolute_error(y_test_raw, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_test))
    r2 = r2_score(y_test_raw, y_pred_test)
    mape = np.mean(np.abs((y_test_raw - y_pred_test) / y_test_raw)) * 100

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")

    # 12. Combine results with dates
    results = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test_raw,
        'Predicted': y_pred_test
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results['Date'],
        y=results['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='royalblue')
    ))

    fig.add_trace(go.Scatter(
        x=results['Date'],
        y=results['Predicted'],
        mode='lines',
        name='Predicted',
        line=dict(color='firebrick')
    ))

    fig.update_layout(
        title='SVR Prediction vs Actual BTC_Close_BTC',
        xaxis_title='Date',
        yaxis_title='BTC_Close_BTC',
        template='plotly_dark'
    )

    return fig