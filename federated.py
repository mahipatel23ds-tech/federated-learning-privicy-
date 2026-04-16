import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import add_differential_privacy_noise
from model import normalize, fraud_rule


# --------------------------------------------------
# CLIENT DATA GENERATION
# --------------------------------------------------

def generate_client_data(seed):

    np.random.seed(seed)

    amount = np.random.uniform(10, 10000, 120)
    time = np.random.randint(0, 24, 120)
    location = np.random.randint(0, 5, 120)
    device = np.random.randint(0, 3, 120)

    X = np.column_stack((amount, time, location, device))

    y = np.array([
        1 if fraud_rule(a, t, d) else 0
        for a, t, d in zip(amount, time, device)
    ])

    return X, y


# --------------------------------------------------
# CLIENT TRAINING
# --------------------------------------------------

def train_client_model(client_id):

    seed = abs(hash(client_id)) % 10000

    X, y = generate_client_data(seed)

    # Use GLOBAL normalization
    X_scaled = normalize(X)

    model = LogisticRegression()

    model.fit(X_scaled, y)

    weights = np.append(
        model.coef_.flatten(),
        model.intercept_
    )

    # Differential Privacy
    noisy_weights = add_differential_privacy_noise(weights)

    return noisy_weights


# --------------------------------------------------
# FEDERATED AVERAGING
# --------------------------------------------------

def federated_averaging(weights_list):

    weights_array = np.array(weights_list)

    avg_weights = np.mean(weights_array, axis=0)

    return avg_weights