import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def encode(seq):
    '''
    Encode DNA sequences as a one-hot numeric array to be used for training.

    :parameter seq: sequence to encode.
    :return: one-hot-encoded array
    '''

    # Define universe of possible input values (Genetic code, 4 bases)
    dna_code = 'TACG'

    # Define a mapping of DNA nucleotides to integers
    char_to_int = dict((c, i) for i, c in enumerate(dna_code))

    # Integer encode DNA sequence
    int_encoded = [char_to_int[char] for char in seq]

    # One hot encode DNA sequence
    onehot_encoded = []

    for value in int_encoded:
        letter = [0 for _ in range(len(dna_code))]
        letter[value] = 1
        onehot_encoded.append(letter)

    return np.array(onehot_encoded)


def ohe_model(ds, seq_col=2, eff_col=3, transform=False, save=False):
    '''
    Train an Extreme Gradient Boost model using One-Hot-Encoding to represent the DNA sequences.
    For instance, use 'ohe_model(ds, transform=True)' for Chari dataset, 'ohe_model(ds, 2, 9)' for DeepSpCas9 and 'ohe_model(ds)' for the remaining ones.

    :parameter ds: dataset to train on.
    :parameter seq_col: # position of column containing 30-nt sequences e.g. 2nd column -> 2.
    :parameter eff_col: # position of column containing efficiencies e.g. 3rd column -> 3.
    :parameter transform: If True, apply square root transformation to all efficiencies (only use for Chari dataset).
    :parameter save: If True, save the trained model to the working directory.
    :return: the trained model.
    '''

    # Encode sequences and define features & labels
    X = ds.iloc[:, seq_col-1].apply(encode)
    X_new = np.stack(X)
    X_train = X_new.reshape(X_new.shape[0], 120)

    Y_train = ds.iloc[:, eff_col-1].values

    # Square root transformation of efficiencies (only for Chari dataset)
    if transform == True:
        Y_train = np.sqrt(Y_train)

    # Initialize and train XGB model
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, Y_train)

    # Return (and save) trained model
    if save == True:
        joblib.dump(model, 'xgb.joblib')
        return model
    else:
        return model
