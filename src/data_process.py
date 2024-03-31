import numpy as np
import pywt
import torch
import wfdb
from imblearn.under_sampling import RandomUnderSampler
from typing import List, Tuple

try:
    from rich import print
except:
    pass


def sigma_delta_encoding(data: np.ndarray, interval_size: int, num_intervals: int) -> torch.Tensor:
    """ 
    Perform sigma-delta encoding on ECG data
        
    Args:
        data (np.ndarray): ECG data with a time step of 300
        interval_size (int): Divide the time interval into `interval_size` parts
        num_intervals (int): Divide the amplitude range into `num_intervals` parts
    
    Returns:
        output_matrix (torch.Tensor): shape(2, interval_size), where the first row represents 
                                      the count of upward threshold crossings and the second row represents 
                                      the count of downward threshold crossings for each interval. 

    """

    assert 300 % interval_size == 0, f"interval_size is {interval_size}, can't be divided by 300"
    
    # Reshape data according to the interval size
    data = data.reshape(interval_size, -1)

    # Create thresholds for sigma-delta encoding, spaced evenly between -2 and 6
    thresholds = torch.linspace(-2, 6, num_intervals+1)[1:-1]

    # Convert numpy array to torch tensor
    data = torch.tensor(data)

    # Expand dimensions of data and thresholds for vectorized comparison
    data_expanded = data.unsqueeze(2)  # Add a new dimension for thresholds comparison
    thresholds_expanded = thresholds.unsqueeze(0).unsqueeze(0) # Expand threshold dimensions

    # Determine crossings: where data crosses thresholds from below (upper_cross) or above (lower_cross)
    upper_cross = (data_expanded[:, :-1] < thresholds_expanded) & (data_expanded[:, 1:] > thresholds_expanded)
    lower_cross = (data_expanded[:, :-1] > thresholds_expanded) & (data_expanded[:, 1:] < thresholds_expanded)

    # Count threshold crossings for each interval
    upper_thresh_counts = upper_cross.sum(dim=1).sum(dim=1)  # Count of upward crossings
    lower_thresh_counts = lower_cross.sum(dim=1).sum(dim=1)  # Count of downward crossings

    # Stack the counts of upper and lower threshold crossings to form the output matrix
    output_matrix = torch.stack([upper_thresh_counts, lower_thresh_counts], dim=0)

    return output_matrix.T

def denoise(data: np.ndarray, wavelet: str='db5', level: int=9) -> np.ndarray:
    """
    Denoise data using wavelet transform.

    Args:
        data (np.ndarray): Input data to be denoised.
        wavelet (str): Wavelet type to use for the transformation. Default is 'db5'.
        level (int): Level of wavelet decomposition. Default is 9.

    Returns:
        np.ndarray: Denoised data.
    """
    
    # Decompose data with wavelet transform
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)

    # Calculate the universal threshold
    detail_coeffs = coeffs[-1]
    median_abs_deviation = np.median(np.abs(detail_coeffs))
    universal_threshold = median_abs_deviation / 0.6745 * np.sqrt(2 * np.log(len(detail_coeffs)))

    # Apply thresholding to detail coefficients
    coeffs[1:] = [pywt.threshold(c, universal_threshold) for c in coeffs[1:]]

    # Reconstruct data from thresholded coefficients
    return pywt.waverec(coeffs=coeffs, wavelet=wavelet)


def z_score_norm(data: np.ndarray) -> np.ndarray:
    """
        Normalize the given data using the Z-score normalization method.
    """
    eps = 1e-6 # Small epsilon to prevent division by zero
    mean = np.mean(data)
    std = np.std(data)

    return (data - mean) / (std + eps)


def read_and_denoise_ecg_data(record_number: str, X_data: List[np.ndarray], Y_data: List[int]) -> None:
    """
    Reads ECG data for a given record number, performs wavelet denoising, and extracts relevant features.

    Args:
        record_number (str): The record number to read ECG data from.
        X_data (List[np.ndarray]): The list to store extracted ECG features.
        Y_data (List[int]): The list to store corresponding labels for ECG features.

    Returns:
        None: The function modifies X_data and Y_data in place.
    """

    ecg_class_set = ['N', 'L', 'R', 'V']
    print(f"Reading No.{record_number} ECG data...")

    # Read ECG data
    record_path = f'Dataset/mit-bih-arrhythmia-database-1.0.0/{record_number}'
    record = wfdb.rdrecord(record_path, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    # Read annotations for R waves
    annotation = wfdb.rdann(record_path, 'atr')
    r_location, r_class = annotation.sample, annotation.symbol

    # Process data, ignoring unstable start and end segments
    for i in range(10, len(r_class) - 5):
        if r_class[i] in ecg_class_set:
            label = ecg_class_set.index(r_class[i])
            x_train = rdata[r_location[i] - 99:r_location[i] + 201]
            X_data.append(x_train)
            Y_data.append(label)


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray]:
    """
        Load ECG dataset, perform preprocessing, and balance classes via undersampling.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of processed feature array X and label array Y.
    """

    number_set = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    data_set, label_set = [], []
    
    # Read and preprocess data for each record
    for number in number_set:
        read_and_denoise_ecg_data(number, data_set, label_set)

    # Perform random undersampling to balance the classes
    rus = RandomUnderSampler(random_state=0)
    x_resampled, y_resampled = rus.fit_resample(data_set, label_set)
    # X_resampled = z_score_norm(X_resampled)

    # Convert to numpy arrays and shuffle
    x_resampled = np.array(x_resampled).reshape(-1, 300, 1)
    y_resampled = np.array(y_resampled)

    # Shuffle the dataset
    indices = np.arange(x_resampled.shape[0])
    np.random.shuffle(indices)
    x = x_resampled[indices]
    y = y_resampled[indices]

    return x, y
