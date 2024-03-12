import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math

def read_signature_file(file_path):
    """
    Reads a single signature file, normalizes the X and Y coordinates, and returns its contents as a pandas DataFrame.
    """
    with open(file_path, 'r') as file:
        # Read the total number of points (the first line)
        total_points = int(file.readline().strip())
        
        # Read each point and store in a list
        points = []
        for _ in range(total_points):
            temp = []
            line = file.readline().strip().split()
            temp.append([float(value) for value in line])

            # Path-Tangent
            path_tangent = math.atan(temp[1]/temp[0])
            temp.append(path_tangent)

            # Path-Velocity
            path_velocity = (temp[0]**2 + temp[1]**2)**0.5
            temp.append(path_velocity)

            # Log-curvature
            log_curvature = math.log(path_velocity/path_tangent)
            temp.append(log_curvature)

            # Acceleration
            acceleration = (path_velocity**2 + (path_velocity*path_tangent)**2)**0.5
            temp.append(acceleration)

            points.append(temp)

    # Convert the list of points into a DataFrame
    columns = ['X-coordinate', 'Y-coordinate', 'Time stamp', 'Button status', 'Azimuth', 'Altitude', 'Pressure', 'Path-tangent', 'Path-velocity', 'Log-curvature', 'Acceleration']
    df = pd.DataFrame(points, columns=columns)
    
    # Normalize the X and Y coordinates
    df['X-coordinate'] = (df['X-coordinate'] - df['X-coordinate'].mean()) / (df['X-coordinate'].max() - df['X-coordinate'].min())
    df['Y-coordinate'] = (df['Y-coordinate'] - df['Y-coordinate'].mean()) / (df['Y-coordinate'].max() - df['Y-coordinate'].min())
    
    return df

def read_all_signatures(base_path='Task2'):
    """
    Reads all signature files for all users and returns a nested dictionary
    with user and signature IDs as keys.
    """
    # Initialize a dictionary to hold all the data
    all_data = {}

    # Loop through the user folders
    for user_id in range(1, 41):  # Assuming 40 users
        user_key = f'U{user_id}'
        all_data[user_key] = {'genuine': [], 'forgery': []}
        
        # Read each signature file for the user
        for sig_id in range(1, 41):  # 20 genuine + 20 forgeries
            file_name = f"{user_key}S{sig_id}.TXT"
            file_path = os.path.join(base_path, file_name)
            
            if os.path.exists(file_path):
                df = read_signature_file(file_path)
                
                # Store the DataFrame in the appropriate category
                if sig_id <= 20:
                    all_data[user_key]['genuine'].append(df)
                else:
                    all_data[user_key]['forgery'].append(df)
            else:
                print(f"File not found: {file_path}")
                
    return all_data

def calculate_average_data_points(all_signatures):
    """
    Calculates the average number of data points for each user's signatures.
    
    :param all_signatures: Nested dictionary containing users' signatures data.
    :return: Dictionary with user IDs as keys and average number of data points as values.
    """
    average_data_points_per_user = {}
    
    for user_key, signatures in all_signatures.items():
        total_data_points = 0
        total_signatures = 0
        
        # Calculate total data points for genuine signatures
        for df in signatures['genuine']:
            total_data_points += len(df)
            total_signatures += 1
        
        # Calculate total data points for forgery signatures
        # for df in signatures['forgery']:
        #     total_data_points += len(df)
        #     total_signatures += 1
        
        # Calculate average data points for the user
        average_data_points = total_data_points / total_signatures
        average_data_points_per_user[user_key] = average_data_points
    
    return average_data_points_per_user

def interpolate_signature(signature_df, target_length):
    """
    Interpolates a signature to a given target length.
    
    :param signature_df: DataFrame containing the signature to be interpolated.
    :param target_length: The target length to interpolate the signature to.
    :return: Interpolated DataFrame of the signature.
    """
    # Assuming linear interpolation on the 'X-coordinate' and 'Y-coordinate'
    # You might need to interpolate other columns depending on your requirements
    x = np.linspace(0, 1, len(signature_df))
    x_new = np.linspace(0, 1, target_length)
    
    interpolated_data = {}
    for column in signature_df.columns:
        y = signature_df[column]
        y_new = np.interp(x_new, x, y)
        interpolated_data[column] = y_new
    
    return pd.DataFrame(interpolated_data)

def interpolate_all_signatures(all_signatures, average_data_points):
    """
    Interpolates all signatures for each user based on the average signature length of that user.
    
    :param all_signatures: Nested dictionary containing users' signatures data.
    :param average_data_points: Dictionary with user IDs as keys and average number of data points as values.
    """
    for user_key, signatures in all_signatures.items():
        target_length = int(average_data_points[user_key])
        
        # Interpolate genuine signatures
        for i, df in enumerate(signatures['genuine']):
            all_signatures[user_key]['genuine'][i] = interpolate_signature(df, target_length)
        
        # Interpolate forgery signatures
        # for i, df in enumerate(signatures['forgery']):
        #     all_signatures[user_key]['forgery'][i] = interpolate_signature(df, target_length)

def calculate_average_user(all_signatures):
    """
    Calculates the mean of the corresponding rows across all genuine and forgery signatures for each user separately.
    
    :param all_signatures: Nested dictionary containing users' signatures data.
    :return: Dictionary with user IDs as keys. Each key maps to another dictionary with 'genuine' and 'forgery' keys, 
             where each value is a DataFrame representing the mean signature.
    """
    user_mean_signatures = {}
    
    for user_key, signatures in all_signatures.items():
        user_mean_signatures[user_key] = {}
        
        for signature_type in ['genuine', 'forgery']:
            signature_list = signatures[signature_type]
            
            # Check if there are signatures to process
            if signature_list:
                # Concatenate all signatures with an additional identifier
                concatenated_df = pd.concat(signature_list, keys=range(len(signature_list)))
                
                # Calculate the mean across all signatures for each row
                mean_signature = concatenated_df.groupby(level=1).mean()
                
                # Store the mean signature in the dictionary under the appropriate type
                user_mean_signatures[user_key][signature_type] = mean_signature
    
    return user_mean_signatures

def perform_dba_on_signatures(mean_signature, signatures):
    """
    Refine the average signature using DTW and the DBA approach with fastdtw.
    
    :param mean_signature: The initial mean signature to refine.
    :param signatures: A list of DataFrame signatures to be averaged.
    :return: The refined average signature as a DataFrame.
    """
    # Initialize the structure to hold the sum of aligned points for averaging.
    num_params = len(mean_signature.columns)
    aligned_sum = np.zeros((len(mean_signature), num_params))
    # Counter for the number of points aligned to each index for averaging.
    aligned_count = np.zeros(len(mean_signature))

    for signature in signatures:
        # Convert DataFrames to numpy arrays for fastdtw
        mean_signature_np = mean_signature.to_numpy()
        signature_np = signature.to_numpy()
        
        # Perform fast DTW between the mean_signature and the current signature
        distance, path = fastdtw(mean_signature_np, signature_np, dist=euclidean)
        
        # Traverse the warping path to align and sum the signatures
        for mean_index, sig_index in path:
            aligned_sum[mean_index] += signature_np[sig_index]
            aligned_count[mean_index] += 1
    
    # Compute the new average signature
    new_mean_signature_np = aligned_sum / aligned_count[:, np.newaxis]  # Normalize by the count

    # Convert the numpy array back to a DataFrame
    new_mean_signature = pd.DataFrame(new_mean_signature_np, columns=mean_signature.columns)
    
    return new_mean_signature

def eb_dba():
    all_signatures = read_all_signatures()
    average_data_points = calculate_average_data_points(all_signatures)
    interpolate_all_signatures(all_signatures, average_data_points)
    user_mean_signatures = calculate_average_user(all_signatures)
    for user_key, signatures in user_mean_signatures.items():
        genuine_mean_signature = signatures['genuine']
        refined_genuine_mean = perform_dba_on_signatures(genuine_mean_signature, all_signatures[user_key]['genuine'])
        # Update the user's mean genuine signature with the refined version.
        user_mean_signatures[user_key]['genuine'] = refined_genuine_mean

    print(user_mean_signatures['U1']['genuine'])   
    

eb_dba()
