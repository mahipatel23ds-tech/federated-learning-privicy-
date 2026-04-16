import numpy as np


# --------------------------------------------------
# DIFFERENTIAL PRIVACY
# --------------------------------------------------

def add_differential_privacy_noise(weights, epsilon=0.05):

    """
    Adds Laplace noise to model weights
    to preserve client privacy.
    """

    noise = np.random.laplace(
        loc=0,
        scale=epsilon,
        size=len(weights)
    )

    noisy_weights = weights + noise

    return noisy_weights