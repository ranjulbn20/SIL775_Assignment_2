import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
import copy
import matplotlib.pyplot as plt

def read_signature_file(file_path):
    with open(file_path, 'r') as file:
        # Read the total number of points (the first line)
        total_points = int(file.readline().strip())

        # Read each point and store in a list
        points = []
        for _ in range(total_points):
            line = file.readline().strip().split()
            temp = [float(value) for value in line]
            points.append(temp)

    # Convert the list of points into a DataFrame
    columns = ['X-coordinate', 'Y-coordinate', 'Time stamp', 'Button status', 'Azimuth',
               'Altitude', 'Pressure']
    df = pd.DataFrame(points, columns=columns)

    # Normalize the X and Y coordinates
    df['X-coordinate'] = (df['X-coordinate'] - df['X-coordinate'].mean()) / \
        (df['X-coordinate'].max() - df['X-coordinate'].min())
    df['Y-coordinate'] = (df['Y-coordinate'] - df['Y-coordinate'].mean()) / \
        (df['Y-coordinate'].max() - df['Y-coordinate'].min())

    small_constant = 1e-9
    # Calculate Path-Tangent, Path-Velocity, Log-Curvature, and Acceleration
    df['Path-tangent'] = np.arctan2(df['Y-coordinate'], df['X-coordinate'])
    df['Path-velocity'] = np.sqrt(df['X-coordinate']
                                  ** 2 + df['Y-coordinate']**2)
    df['Log-curvature'] = np.log(abs(df['Path-velocity'] /
                                 (df['Path-tangent']+small_constant)))
    df['Acceleration'] = np.sqrt(
        df['Path-velocity']**2 + (df['Path-velocity'] * df['Path-tangent'])**2)

    df = df.drop(
        columns=['Time stamp', 'Button status', 'Azimuth', 'Altitude'])
    return df


def read_all_signatures(base_path='Task2'):
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
    x = np.linspace(0, 1, len(signature_df))
    x_new = np.linspace(0, 1, target_length)

    interpolated_data = {}
    for column in signature_df.columns:
        y = signature_df[column]
        y_new = np.interp(x_new, x, y)
        interpolated_data[column] = y_new

    return pd.DataFrame(interpolated_data)


def interpolate_all_signatures(all_signatures, average_data_points):
    for user_key, signatures in all_signatures.items():
        target_length = int(average_data_points[user_key])

        # Interpolate genuine signatures
        for i, df in enumerate(signatures['genuine']):
            all_signatures[user_key]['genuine'][i] = interpolate_signature(
                df, target_length)

        # Interpolate forgery signatures
        # for i, df in enumerate(signatures['forgery']):
        #     all_signatures[user_key]['forgery'][i] = interpolate_signature(df, target_length)


def calculate_average_user(all_signatures):
    user_mean_signatures = {}

    for user_key, signatures in all_signatures.items():
        user_mean_signatures[user_key] = {}

        for signature_type in ['genuine', 'forgery']:
            signature_list = signatures[signature_type]

            # Check if there are signatures to process
            if signature_list:
                # Concatenate all signatures with an additional identifier
                concatenated_df = pd.concat(
                    signature_list, keys=range(len(signature_list)))

                # Calculate the mean across all signatures for each row
                mean_signature = concatenated_df.groupby(level=1).mean()

                # Store the mean signature in the dictionary under the appropriate type
                user_mean_signatures[user_key][signature_type] = mean_signature

    return user_mean_signatures


def perform_dba_on_signatures(mean_signature, signatures):
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
        distance, path = fastdtw(
            mean_signature_np, signature_np, dist=euclidean)

        # Traverse the warping path to align and sum the signatures
        for mean_index, sig_index in path:
            aligned_sum[mean_index] += signature_np[sig_index]
            aligned_count[mean_index] += 1

    # Compute the new average signature
    new_mean_signature_np = aligned_sum / \
        aligned_count[:, np.newaxis]  # Normalize by the count

    # Convert the numpy array back to a DataFrame
    new_mean_signature = pd.DataFrame(
        new_mean_signature_np, columns=mean_signature.columns)

    return new_mean_signature


def calculate_accuracy_precision_and_recall(user_key, signatures, user_mean_signatures, threshold):
    correct_classifications = 0
    total_classifications = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for signature in signatures['genuine']:
        mean_signature_np = user_mean_signatures[user_key]['genuine'].to_numpy(
        )
        signature_np = signature.to_numpy()
        distance, _ = fastdtw(mean_signature_np, signature_np, dist=euclidean)
        if distance <= threshold:
            correct_classifications += 1
            true_positives += 1
        else:
            false_negatives += 1
        total_classifications += 1

    # Test forgery signatures
    for signature in signatures['forgery']:
        mean_signature_np = user_mean_signatures[user_key]['genuine'].to_numpy(
        )
        signature_np = signature.to_numpy()
        distance, _ = fastdtw(mean_signature_np, signature_np, dist=euclidean)
        if distance > threshold:
            correct_classifications += 1
            true_negatives += 1
        else:
            false_positives += 1
        total_classifications += 1

    accuracy = (correct_classifications / total_classifications) * 100
    precision = (true_positives / (true_positives + false_positives)
                 ) * 100 
    recall = (true_positives / (true_positives + false_negatives)) * \
        100 
    far = (false_positives / (false_positives + true_negatives)) * \
        100 
    frr = (false_negatives / (false_negatives + true_positives)) * \
        100 

    return accuracy, precision, recall, far, frr

def calculate_err_for_user_and_plot(user_key, signatures, user_mean_signatures, threshold):
    threshold_values = np.linspace(start=0.5*threshold, stop=1.5*threshold, num=500)
    far_values = []
    frr_values = []

    for threshold in threshold_values:
        _, _, _, far, frr = calculate_accuracy_precision_and_recall(
            user_key, signatures, user_mean_signatures, threshold)
        far_values.append(far)
        frr_values.append(frr)

    index_of_eer = np.argmin(np.abs(np.subtract(far_values, frr_values)))
    eer_threshold = threshold_values[index_of_eer]
    eer_value = (far_values[index_of_eer] + frr_values[index_of_eer]) / 2
    print(f"EER Threshold: {eer_threshold}, EER Value: {eer_value}%")

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_values, far_values, label='FAR')
    plt.plot(threshold_values, frr_values, label='FRR')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate (%)')
    plt.title('EER vs Thresholds')
    plt.legend()
    plt.grid(True)
    plt.show()

def eb_dba():
    all_signatures = read_all_signatures()
    average_data_points = calculate_average_data_points(all_signatures)
    all_signatures_copy = copy.deepcopy(all_signatures)
    interpolate_all_signatures(all_signatures, average_data_points)
    user_mean_signatures = calculate_average_user(all_signatures)
    for user_key, signatures in user_mean_signatures.items():
        genuine_mean_signature = signatures['genuine']
        refined_genuine_mean = perform_dba_on_signatures(
            genuine_mean_signature, all_signatures[user_key]['genuine'])
        # Update the user's mean genuine signature with the refined version.
        user_mean_signatures[user_key]['genuine'] = refined_genuine_mean

    users_thresholds = {}
    thresholds = []
    for user_key, signatures in all_signatures_copy.items():
        dist_genuine = []
        for signature in signatures['genuine']:
            mean_signature_np = user_mean_signatures[user_key]['genuine'].to_numpy(
            )
            signature_np = signature.to_numpy()
            distance, _ = fastdtw(
                mean_signature_np, signature_np, dist=euclidean)
            dist_genuine.append(distance)

        dist_forgery = []
        for signature in signatures['forgery']:
            mean_signature_np = user_mean_signatures[user_key]['genuine'].to_numpy(
            )
            signature_np = signature.to_numpy()
            distance, _ = fastdtw(
                mean_signature_np, signature_np, dist=euclidean)
            dist_forgery.append(distance)

        threshold = (max(dist_genuine) + min(dist_forgery))//2
        users_thresholds[user_key] = threshold
        thresholds.append(threshold)

    far_list = []
    frr_list = []
    user_metrics = {}
    with open('output.txt', 'w') as file:
        for user_key, signatures in all_signatures_copy.items():
            accuracy, precision, recall, far, frr = calculate_accuracy_precision_and_recall(
                user_key, signatures, user_mean_signatures, users_thresholds[user_key])
            far_list.append(far)
            frr_list.append(frr)
            user_metrics[user_key] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'FAR': far,
                'FRR': frr
            }
            print(
                f"User {user_key}: Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}, FAR = {far}, FRR = {frr}",
                file=file)

        total_accuracy = sum(user['accuracy'] for user in user_metrics.values())
        total_precision = sum(user['precision'] for user in user_metrics.values())
        total_recall = sum(user['recall'] for user in user_metrics.values())
        number_of_users = len(user_metrics)
        overall_accuracy = total_accuracy / number_of_users
        overall_precision = total_precision / number_of_users
        overall_recall = total_recall / number_of_users
        print(f"Overall Accuracy: {overall_accuracy}%", file=file)
        print(f"Overall Precision: {overall_precision}%", file=file)
        print(f"Overall Recall: {overall_recall}%", file=file)
    
    user_key = 'U1'
    calculate_err_for_user_and_plot(user_key, all_signatures_copy[user_key], user_mean_signatures, users_thresholds[user_key])
    


eb_dba()
