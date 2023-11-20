import pandas as pd
import numpy as np
import math

# Read the csv file of the face data
face = pd.read_csv("./penstate_data/download/qls.csv")

print(f"Number of people: {face.shape[0]}")
# Number of people: 1078

num_cols = face.shape[1] - 1 # remove the first ID column
print(f"Number of coordinates from each face: {num_cols}")
# Number of coordinates from each face: 20370

num_coordinate_col = num_cols//3
print(f"Number of points (x, y, z) collected from each face: {num_coordinate_col}")
# Number of points (x, y, z) collected from each face: 6790# For one point, there are three coodrinates, however they are not stored together
# All x coordinates are stored in the first 6790 columns,
# All y coordinates are stored in the second 6790 columns,
# All z coordinates are stored in the third 6790 columns

# convert to numpy array
np_face = np.array(face)


# Pick 65 points on the face
face_indices = [2570, 2569, 2564, 3675, 2582,
                 1445, 3848, 1427, 1432, 1433,
                 2428, 2451, 2495, 2471, 3638, 2276, 2355, 2359,
                 3835, 1292, 1344, 1216, 1154, 999, 991, 4046,
                 3704, 3553, 3561, 3501, 3564,
                 2747, 1613, 1392, 3527, 471, 480, 1611,
                 3797, 2864, 2811, 3543, 1694, 1749,
                 3920, 2881, 2905, 1802, 1774, 3503,
                 3515, 3502, 3401, 3399, 3393,
                 3385, 1962, 3381, 3063, 3505,
                 3595, 3581, 3577, 2023, 567]

# Different AM chosen
measurement_info = [
            # 0 - 24
            ['dist', 10, 14],
            ['dist', 18, 22],
            ['dist', 14, 18],
            ['dist', 10, 22],
            ['dist', 4, 5],
            ['dist', 0, 9],
            ['dist', 2, 7],
            ['dist', 58, 60],
            ['dist', 57, 61],
            ['dist', 56, 62],
            ['dist', 55, 63],
            ['dist', 54, 64],
            ['dist', 31, 37], ##
            ['dist', 32, 36], ##
            ['dist', 33, 35], #
            ['dist', 48, 46], ##
            ['dist', 40, 42], ##
            ['dist', 54, 53],
            ['dist', 53, 64],
            ['dist', 38, 44],
            ['dist', 48, 46], ##
            ['dist', 39, 43], #
            ['dist', 41, 47], #
            ['dist', 12, 16],
            ['dist', 20, 24],

            # 25 - 49
            ['dist', 59, 27],
            ['dist', 59, 30],
            ['dist', 59, 34],
            ['dist', 59, 50],
            ['dist', 59, 51],
            ['dist', 59, 53],
            ['dist', 27, 30],
            ['dist', 26, 50],
            ['dist', 30, 34],
            ['dist', 30, 51],
            ['dist', 30, 52],
            ['dist', 30, 53],
            ['dist', 34, 50],
            ['dist', 41, 47], #
            ['dist', 51, 52],
            ['dist', 52, 53],
            ['dist', 50, 53],
            ['dist', 54, 55],
            ['dist', 55, 56],
            ['dist', 56, 57],
            ['dist', 57, 58],
            ['dist', 60, 61],
            ['dist', 61, 62],
            ['dist', 62, 63],
            ['dist', 63, 64],

            # 50 - 63
            ['prop', 31, 37, 27, 30], ##
            ['prop', 32, 36, 27, 30], ##
            ['prop', 57, 61, 27, 30],
            ['prop', 56, 62, 27, 30],
            ['prop', 55, 63, 27, 30],
            ['prop', 54, 64, 27, 30],
            ['prop', 38, 44, 27, 30],
            ['prop', 31, 37, 59, 53], ##
            ['prop', 32, 36, 59, 53], #
            ['prop', 57, 61, 59, 53],
            ['prop', 56, 62, 59, 53],
            ['prop', 55, 63, 59, 53],
            ['prop', 54, 64, 59, 53],
            ['prop', 38, 44, 59, 53],

            # 64 - 77
            ['prop', 58, 60, 57, 61],
            ['prop', 57, 61, 56, 62],
            ['prop', 56, 62, 55, 63],
            ['prop', 55, 63, 54, 64],
            ['prop', 57, 61, 31, 37], #
            ['prop', 56, 62, 31, 37], #
            ['prop', 55, 63, 31, 37], #
            ['prop', 58, 60, 31, 37], #
            ['prop', 54, 64, 31, 37], #
            ['prop', 38, 44, 31, 37],
            ['prop', 59, 53, 27, 30],
            ['prop', 51, 52, 52, 53],
            ['prop', 50, 51, 50, 53],
            ['prop', 27, 30, 27, 53],

            # 78 - 86
            ['angle', 56, 57, 58], 
            ['angle', 55, 56, 57], 
            ['angle', 54, 55, 56],
            ['angle', 53, 54, 55],
            ['angle', 64, 53, 54],
            ['angle', 63, 64, 53],
            ['angle', 62, 63, 64],
            ['angle', 61, 62, 63],
            ['angle', 60, 61, 62],

            # 87 - 95
            ['angle', 31, 29, 37], ##
            ['angle', 31, 30, 37], ##
            ['angle', 27, 30, 34],
            ['angle', 51, 52, 53],
            ['angle', 27, 30, 31],
            ['angle', 37, 30, 27],
            ['angle', 37, 50, 31],
            ['angle', 63, 53, 55],
            ['angle', 29, 30, 34],
    
#              # 78 - 86
#             ['angle', 56, 57, 58, -0.1011, -0.3078,  0.9344], 
#             ['angle', 55, 56, 57, -0.9009, -0.1629,  0.3678], 
#             ['angle', 54, 55, 56, -0.7863, -0.1348,  0.5986],
#             ['angle', 53, 54, 55, -0.3654,  0.5084,  0.7771],
#             ['angle', 64, 53, 54, -0.0262,  0.9601,  0.2719],
#             ['angle', 63, 64, 53,  0.2780,  0.6064,  0.7425],
#             ['angle', 62, 63, 64,  0.8007, -0.1164,  0.5832],
#             ['angle', 61, 62, 63,  0.9095, -0.1439,  0.3239],
#             ['angle', 60, 61, 62,  0.4377, -0.2226,  0.8568],

#             # 87 - 95
#             ['angle', 31, 29, 37, -0.0054, -0.6474,  0.7596], ##
#             ['angle', 31, 30, 37, -0.0098, -0.9611,  0.2697], ##
#             ['angle', 27, 30, 34, -0.9997, -0.0218, -0.0046],
#             ['angle', 51, 52, 53, -0.9893,  0.0432, -0.0409],
#             ['angle', 27, 30, 31, -0.7108,  0.4236,  0.5553],
#             ['angle', 37, 30, 27,  0.7083,  0.4411,  0.5448],
#             ['angle', 37, 50, 31,  0.0048,  0.2632,  0.9614],
#             ['angle', 63, 53, 55,  0.0039,  0.8142,  0.5791],
#             ['angle', 29, 30, 34, -0.9987, -0.0466,  0.0199],
    
        ]


def get_distance(face, idx1, idx2):
    """
    This function calculates the Euclidean distance between two points for every face.

    Parameters:
    face (numpy.ndarray): The face data.
    idx1 (int): The index of the first point.
    idx2 (int): The index of the second point.

    Returns:
    numpy.ndarray: The Euclidean distance between two points for every face.
    """
    # For one point, there are three coodrinates, however they are not stored together
    # All x coordinates are stored in the first 6790 columns,
    # All y coordinates are stored in the second 6790 columns,
    # All z coordinates are stored in the third 6790 columns
    num_cordinate_col = face.shape[1] // 3

    # Extract the x, y, z coordinate of idx1 from the orginal dataset
    cordinate1 = face[:, [idx1, idx1 + num_cordinate_col, idx1 + 2 * num_cordinate_col]]
    
    # Extract the x, y, z coordinate of idx2 from the orginal dataset
    cordinate2 = face[:, [idx2, idx2 + num_cordinate_col, idx2 + 2 * num_cordinate_col]]

    # Calculate the Euclidean distance between two points for every face
    # The final shape is (1078,), which is the number of faces
    distance = np.linalg.norm(cordinate2 - cordinate1, axis=0)
    return distance

# Example
print("index[10], index[14] = ", face_indices[10], face_indices[14])
print(get_distance(np_face, face_indices[10], face_indices[14]))


def get_proportion(target, idx1, idx2, idx3, idx4):
    """
    This function calculates the proportion of the distances between two pairs of points for every face.

    Parameters:
    target (numpy.ndarray): The face data.
    idx1, idx2 (int): The indices of the first pair of points.
    idx3, idx4 (int): The indices of the second pair of points.

    Returns:
    numpy.ndarray: The proportion of the distances between two pairs of points for every face.
    """
    dist1 = get_distance(target, idx1, idx2)
    dist2 = get_distance(target, idx3, idx4)
    return dist1 / dist2

# Example
print("index[10], index[14], index[18], index[22] = ", face_indices[10], face_indices[14], face_indices[18], face_indices[22])
print(get_proportion(np_face, face_indices[10], face_indices[14], face_indices[18], face_indices[22]))


def get_angle(face, idx1, idx2, idx3):
    """
    This function calculates the angle between three points for every face.

    Parameters:
    face (numpy.ndarray): The face data.
    idx1, idx2, idx3 (int): The indices of the three points.

    Returns:
    numpy.ndarray: The angle between three points for every face in degrees.
    """

    # For one point, there are three coodrinates, however they are not stored together
    # All x coordinates are stored in the first 6790 columns,
    # All y coordinates are stored in the second 6790 columns,
    # All z coordinates are stored in the third 6790 columns
    num_cordinate_col = face.shape[1] // 3

    # Extract the x, y, z coordinate of idx1, idx2, idx3 from the orginal dataset
    cordinate1 = face[:, [idx1, idx1 + num_cordinate_col, idx1 + 2 * num_cordinate_col]]
    cordinate2 = face[:, [idx2, idx2 + num_cordinate_col, idx2 + 2 * num_cordinate_col]]
    cordinate3 = face[:, [idx3, idx3 + num_cordinate_col, idx3 + 2 * num_cordinate_col]]
    
    # get vectors
    v12, v32 = cordinate1 - cordinate2, cordinate3 - cordinate2
    
    # get dot product of two coordinate
    dot_prod = np.sum(v12 * v32, axis=1)
    
    # get magnitude(distance)
    mag12, mag32 = np.linalg.norm(v12, axis=1), np.linalg.norm(v32, axis=1)
    
    # get cos
    cos_ = dot_prod / mag12 / mag32
    
    # get radius
    angle = np.arccos(cos_)
    
    # convert degree
    # Basically doing angle <- angle mod 360
    ang_deg = np.degrees(angle) % 360
    return ang_deg

# Example
print("index[10], index[14], index[18] = ", face_indices[10], face_indices[14], face_indices[18])
print(get_angle(np_face, face_indices[10], face_indices[14], face_indices[18]))
get_angle(np_face, 1, 4, 330)


def calculate_measurement(face_data, measurement_info):
    """
    This function calculates the measurement based on the type of measurement and indices provided.

    Parameters:
    face_data (numpy.ndarray): The face data.
    measurement_info (list): A list containing the type of measurement and indices for the measurement.

    Returns:
    numpy.ndarray: The calculated measurement.
    """
    measurement_type = measurement_info[0]
    index1, index2 = face_indices[measurement_info[1]], face_indices[measurement_info[2]]
    calculated_measurement = None

    if measurement_type == 'dist':
        calculated_measurement = get_distance(face_data, index1, index2)
    elif measurement_type == 'prop':
        index3, index4 = face_indices[measurement_info[3]], face_indices[measurement_info[4]]
        calculated_measurement = get_proportion(face_data, index1, index2, index3, index4)
    elif measurement_type == 'angle':
        index3 = face_indices[measurement_info[3]]
        calculated_measurement = get_angle(face_data, index1, index2, index3)
    else:
        raise ValueError('unknown type {}'.format(measurement_type))

    return calculated_measurement

# Test
print("measurement_info:", measurement_info[0])
print("measurement_info:", measurement_info[67])
print("measurement_info:", measurement_info[90])
calculate_measurement(np_face, measurement_info[2])

# form the new csv of AM column
np_face = np.array(face)
AM_Unnormalized = [calculate_measurement(np_face, info) for info in measurement_info]

AM_Unnormalized_df = pd.DataFrame(AM_Unnormalized).T

AM_Unnormalized_df.columns += 1 # Column start from 1
AM_OG = AM_Unnormalized_df.copy()

# Insert ID back
AM_Unnormalized_df.insert(0, "ID", face["ID"])
# AM_Unnormalized_df

# Normalize the AM
AM_Normalized_df = AM_Unnormalized_df.drop(columns=["ID"])
AM_Normalized_df = (AM_Normalized_df - AM_Normalized_df.mean()) / AM_Normalized_df.std()
AM_Normalized_df.insert(0, "ID", face["ID"])
# AM_Normalized_df

# Export csv
# AM_Normalized_df.to_csv("./AMs_final.csv", index=False)
# AM_Normalized_df


