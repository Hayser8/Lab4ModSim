import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import numpy as np


def load_data():
    """Returns synthetic discrete-time and continuous-time simulation outputs"""
    np.random.seed(42)  # For reproducibility
    
    # Discrete-time data (weekly reporting)
    discrete = {
        'timestamps': np.arange(0, 21),  # Daily timestamps (integer days)
        'infections': [10, 12, 8, 9, 11, 15, 120,  # Weekly spike
                       9, 11, 7, 8, 10, 125,       # Next week
                       8, 10, 6, 9, 12, 118],      # Final week
        'agent_data': pd.DataFrame({
            'age': np.random.choice(['0-18','19-65','65+'], 1000),
            'occupation': np.random.choice(['healthcare','education','other'], 1000),
            'vaccinated': np.random.choice([True, False], 1000, p=[0.6, 0.4])
        })
    }
    
    # Continuous-time data (event-driven)
    t_continuous = np.linspace(0, 21, 500)
    outbreaks = (80 * np.exp(-(t_continuous-3.5)**2/1.5) + 
                100 * np.exp(-(t_continuous-8.2)**2/2) + 
                90 * np.exp(-(t_continuous-14.1)**2/1.8))
    background = 10 * np.sin(0.3*t_continuous) + 15
    
    continuous = {
        'timestamps': t_continuous,
        'infections': outbreaks + background + np.random.normal(0, 3, 500),
        'agent_data': pd.DataFrame({
            'age': np.random.choice(['0-18','19-65','65+'], 1000),
            'mobility': np.random.gamma(2, 1.5, 1000),  # Continuous trait
            'vaccinated': np.random.choice([True, False], 1000, p=[0.6, 0.4])
        })
    }
    
    return {'discrete': discrete, 'continuous': continuous}