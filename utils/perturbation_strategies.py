import numpy as np
import pandas as pd

def label_fliping(X, y, flip_ratio):
    flipped_y = y.copy()
    random_indices = np.random.choice(flipped_y.index, size=int(flip_ratio*flipped_y.shape[0]), replace=False)
    flipped_y.loc[random_indices] = 1 - flipped_y.loc[random_indices]
    
    return X, flipped_y

def feature_noise(X, y, noise_ratio):
    """
    Adds noise to a dataframe based on the distribution of each column.

    Parameters:
    - X (pd.DataFrame): The original dataframe.
    - noise_ratio (float): Scaling factor for the magnitude of the noise.

    Returns:
    - The original dataframe with added noise.
    - y
    """

    stds = X.std()
    
    noise = pd.DataFrame(
        {col: np.random.normal(loc=0, scale=noise_ratio * stds[col], size=len(X)) for col in X.columns},
        index=X.index
    )

    noisy_X = X + noise
    
    return noisy_X, y

def feature_corruption(X, y, corrupt_ratio):

    """
    Sets a certain percentage of data for each feature to 0 to corrupt data

    Parameters:
    - X (pd.DataFrame): The original dataframe.
    - corrupt_ratio (float): Scaling factor for the magnitude of the corruption.

    Returns:
    - corrupted dataframe.
    - y
    """


    corrupted_X = X.copy()

    for column in corrupted_X.columns:
        random_indices = np.random.choice(corrupted_X.index, size=int(corrupt_ratio*corrupted_X[column].shape[0]), replace=False)
        corrupted_X.loc[random_indices,column] = 0
    
    return corrupted_X, y