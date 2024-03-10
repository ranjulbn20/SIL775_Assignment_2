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

def read_all_signatures(base_path='signatures'):
    """
    Reads all signature files for all users and returns a nested dictionary
    with user and signature IDs as keys.
    """
    # Initialize a dictionary to hold all the data
    all_data = {}

    # Loop through the user folders
    for user_id in range(1, 41):  # Assuming 40 users
        user_key = f'USER{user_id}'
        all_data[user_key] = {'genuine': [], 'forgery': []}
        
        # Read each signature file for the user
        for sig_id in range(1, 41):  # 20 genuine + 20 forgeries
            file_name = f"{user_key}_{sig_id}.txt"
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

# Example usage
all_signatures = read_all_signatures()

# Accessing the data
# For example, to get the first genuine signature of USER1:
print(all_signatures['USER1']['genuine'][0])
