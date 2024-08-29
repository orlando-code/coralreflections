
try:
    from importlib import resources
    resource_dir = resources.files('reflectance') / 'resources'
except:
    from pathlib import Path
    resource_dir = Path(__file__).resolve().parent / 'resources'

# import pickle
import numpy as np
import pandas as pd

# read in first AOP model (arbitrary choice)
f_AOP_model = resource_dir / 'AOP_models_Group_1.txt'
with open(f_AOP_model, 'r') as f:
    start_found = False
    skiprows = 0
    while not start_found:
        line = f.readline()
        if line.startswith('wl,'):
            start_found = True
        else:
            skiprows += 1

# read in wavelengths as df
AOP_model = pd.read_csv(f_AOP_model, skiprows=skiprows - 1).set_index('wl')
AOP_model.columns = ['Kd_m', 'Kd_c', 'bb_m', 'bb_c']

# def sub_surface_reflectance(wv, bb, K, H, Rb):
#     sub = AOP_model.loc[wv]
#     bb_lambda = bb * sub.loc[wv, 'bb_m'] + sub.loc[wv, 'bb_c']
#     K_lambda = 2 * K * sub.loc[wv, 'Kd_m'] + sub.loc[wv, 'Kd_c']
#     return bb_lambda / K_lambda + (Rb - bb_lambda / K_lambda) * np.exp(-K_lambda * H)

def sub_surface_reflectance(wv, bb, K, H, Rb, bb_m, bb_c, Kd_m, Kd_c):
    """Radiative transfer model for sub-surface reflectance.
    bb_lambda and K_lambda are calculated as a function of wavelength using the AOP model.
    Characterised by (fixed) coefficient and intercept from AOP model, with a scaling factor 
    set during optimisation.
    Types: arrays/pd.Series
    """
    bb_lambda = bb * bb_m + bb_c
    K_lambda = 2 * K * Kd_m + Kd_c
    return bb_lambda / K_lambda + (Rb - bb_lambda / K_lambda) * np.exp(-K_lambda * H)


# No longer using splines because...
# load splines
# Kd_splines = {}
# bb_splines = {}
# for f in resource_dir.glob('*.pkl'):
#     name = f.stem.split('_')[1]
#     with open(f, 'rb') as file:
#         if 'Kd' in f.stem:
#             Kd_splines[name] = pickle.load(file)
#         else:
#             bb_splines[name] = pickle.load(file)


# def sub_surface_reflectance(wv, bb, K, H, Rb, option='Group1'):
#     bb_lambda = bb * bb_splines[option](wv)
#     K_lambda = K * Kd_splines[option](wv)
    
#     return bb_lambda / K_lambda + (Rb - bb_lambda / K_lambda) * np.exp(-K_lambda * H)

