import numpy as np
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# GLOBAL MODEL
# --------------------------------------------------

global_model = LogisticRegression()

# Global scaling parameters (shared with clients)
GLOBAL_MEAN = None
GLOBAL_STD = None


# --------------------------------------------------
# FRAUD LOGIC
# --------------------------------------------------

def fraud_rule(amount, time, device):

    return ((amount > 5000 and time > 20) or (device == 2))


# --------------------------------------------------
# SYNTHETIC DATA GENERATION
# --------------------------------------------------

def generate_synthetic_data(size=500):

    amount = np.random.uniform(10, 10000, size)
    time = np.random.randint(0, 24, size)
    location = np.random.randint(0, 5, size)
    device = np.random.randint(0, 3, size)

    X = np.column_stack((amount, time, location, device))

    y = np.array([
        1 if fraud_rule(a, t, d) else 0
        for a, t, d in zip(amount, time, device)
    ])

    return X, y


# --------------------------------------------------
# GLOBAL NORMALIZATION
# --------------------------------------------------

def compute_global_scaling(X):

    global GLOBAL_MEAN, GLOBAL_STD

    GLOBAL_MEAN = np.mean(X, axis=0)
    GLOBAL_STD = np.std(X, axis=0) + 1e-8


def normalize(X):

    return (X - GLOBAL_MEAN) / GLOBAL_STD


# --------------------------------------------------
# INITIALIZE GLOBAL MODEL
# --------------------------------------------------

X, y = generate_synthetic_data()

compute_global_scaling(X)

X_scaled = normalize(X)

global_model.fit(X_scaled, y)


# --------------------------------------------------
# UPDATE GLOBAL MODEL
# --------------------------------------------------

def update_global_model(weights):

    global global_model

    global_model.coef_ = np.array(weights[:-1]).reshape(1, -1)
    global_model.intercept_ = np.array([weights[-1]])


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_transaction(amount, time, location, device):

    X = np.array([[amount, time, location, device]])

    X_scaled = normalize(X)

    prob = global_model.predict_proba(X_scaled)[0][1]

    label = "Fraud" if prob > 0.5 else "Not Fraud"

    return prob, label