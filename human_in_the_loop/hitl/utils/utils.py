import numpy as np

# def calculate_percentiles(array: np.ndarray) -> np.ndarray:
#     """Calculate percentiles for the given array."""
#     xmin = np.min(array)
#     xmax = np.max(array)
    
#     percent_of_range = np.interp(array, (xmin, xmax), (0, 100))
#     print(percent_of_range)
#     return percent_of_range

def calculate_percentiles(array: np.ndarray) -> np.ndarray:
    percentiles = np.searchsorted(np.sort(array), array, side='right') / len(array)
    return percentiles 