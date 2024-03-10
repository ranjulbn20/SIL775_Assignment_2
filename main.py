import os
import pandas as pd

def read_signature_file(file_path):
    """
    Reads a single signature file and returns its contents as a pandas DataFrame.
    """
    with open(file_path, 'r') as file:
        # Read the total number of points (the first line)
        total_points = int(file.readline().strip())
        
        # Read each point and store in a list
        points = []
        for _ in range(total_points):
            line = file.readline().strip().split()
            points.append([float(value) for value in line])
            
    # Convert the list of points into a DataFrame
    columns = ['X-coordinate', 'Y-coordinate', 'Time stamp', 'Button status', 'Azimuth', 'Altitude', 'Pressure']
    df = pd.DataFrame(points, columns=columns)
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
            file_name = f"{user_key}S{sig_id}.txt"
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

def calculate_mean_coordinates(all_signatures):
    """
    Calculates the mean of the X and Y coordinates for each signature.
    """
    mean_coordinates = {}

    for user_key, signatures in all_signatures.items():
        mean_coordinates[user_key] = {'genuine': [], 'forgery': []}
        
        for category in ['genuine', 'forgery']:
            for df in signatures[category]:
                mean_x = df['X-coordinate'].mean()
                mean_y = df['Y-coordinate'].mean()
                mean_coordinates[user_key][category].append((mean_x, mean_y))
    
    return mean_coordinates

def eb_dba():
    all_signatures = read_all_signatures()
    mean_coordinates = calculate_mean_coordinates(all_signatures)
    # for user_key, categories in mean_coordinates.items():
    #     print(f"{user_key}:")
    #     for category, means in categories.items():
    #         print(f"  {category}:")
    #         for mean in means:
    #             print(f"    Mean X: {mean[0]}, Mean Y: {mean[1]}")

eb_dba()
