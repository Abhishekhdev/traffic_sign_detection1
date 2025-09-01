# import cv2
# import os
# import numpy as np
# import pickle
# from glob import glob

# def preprocess_data_svm(dataset_dir, img_size=(100, 100)):
#     """
#     Preprocess the dataset for SVM: resize, flatten, and label images.
#     """
#     data = []
#     labels = []
#     class_labels = {}
#     label_index = 0

#     # Traverse the dataset directory and process each class
#     for class_name in sorted(os.listdir(dataset_dir)):
#         class_dir = os.path.join(dataset_dir, class_name)
#         if os.path.isdir(class_dir):
#             class_labels[class_name] = label_index
#             print(f"Processing class: {class_name} (Label: {label_index})")

#             for img_name in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, img_name)
#                 img = cv2.imread(img_path)

#                 if img is not None:
#                     # Resize and flatten the image
#                     img_resized = cv2.resize(img, img_size)
#                     img_flattened = img_resized.flatten()
#                     data.append(img_flattened)
#                     labels.append(label_index)

#             label_index += 1

#     data = np.array(data)
#     labels = np.array(labels)

#     print(f"Data shape: {data.shape}")
#     print(f"Labels shape: {labels.shape}")
#     print(f"Class labels: {class_labels}")

#     # Save the preprocessed data
#     with open("preprocessed_data.pkl", "wb") as f:
#         pickle.dump((data, labels, class_labels), f)

#     return data, labels, class_labels

# # Preprocess the dataset located in "/content/Training"
# dataset_dir = "/content/Training"  # Change this to your directory path
# data, labels, class_labels = preprocess_data_svm(dataset_dir)














# import cv2
# import os
# import pickle

# import numpy as np

# def preprocess_data(dataset_dir, img_size=(100, 100)):
#     data = []
#     labels = []
#     class_labels = {}

#     for idx, class_name in enumerate(os.listdir(dataset_dir)):
#         class_dir = os.path.join(dataset_dir, class_name)
#         if os.path.isdir(class_dir):
#             class_labels[class_name] = idx
#             print(f"Processing class: {class_name} (Label: {idx})")

#             for img_name in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, img_name)
#                 img = cv2.imread(img_path)

#                 if img is not None:
#                     # Resize the image to the target size
#                     img_resized = cv2.resize(img, img_size)

#                     # Append the resized image and corresponding label
#                     data.append(img_resized)
#                     labels.append(idx)

#                     print(f"Loaded image {img_name} for class {class_name}")
#                 else:
#                     print(f"Failed to load image {img_name} for class {class_name}")

#     data = np.array(data)
#     labels = np.array(labels)

#     # Check if data and labels are populated correctly
#     print(f"Data shape: {data.shape}")
#     print(f"Labels shape: {labels.shape}")
#     print(f"Class labels: {class_labels}")

#     # Save the preprocessed data
#     with open("preprocessed_data.pkl", "wb") as f:
#         pickle.dump((data, labels, class_labels), f)

# # Example call to the function
# preprocess_data("traffic_signs_dataset")
