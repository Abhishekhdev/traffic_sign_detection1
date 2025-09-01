# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# # Train the SVM model
# svm_model = SVC(kernel='linear', probability=True, random_state=42)  # You can use 'rbf' kernel for non-linear data
# print("Training SVM model...")
# svm_model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = svm_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # Display classification report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Save the trained SVM model and class labels to a .pkl file
# with open('traffic_sign_svm_model.pkl', 'wb') as f:
#     pickle.dump((svm_model, class_labels), f)

# print("SVM model saved as traffic_sign_svm_model.pkl")













# # import pickle
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.svm import SVC
# # from sklearn.metrics import accuracy_score, classification_report

# # def train_model():
# #     """
# #     Train an SVM model using preprocessed traffic sign data, evaluate its performance,
# #     and save the trained model with class labels.
# #     """
# #     # Load preprocessed data
# #     try:
# #         with open("preprocessed_data.pkl", "rb") as f:
# #             data, labels, class_labels = pickle.load(f)
# #         print(f"Data shape: {data.shape}")
# #         print(f"Labels shape: {labels.shape}")
# #         print(f"Class labels: {class_labels}")
# #     except FileNotFoundError:
# #         print("Error: preprocessed_data.pkl not found. Please preprocess the data first.")
# #         return
# #     except Exception as e:
# #         print(f"Error loading preprocessed data: {e}")
# #         return

# #     # Check if data is empty
# #     if data.size == 0 or labels.size == 0:
# #         print("Error: Data or labels are empty. Please preprocess the data correctly.")
# #         return

# #     # Flatten the images (convert 2D images to 1D feature vectors)
# #     data = data.reshape(data.shape[0], -1)  # (num_samples, height * width * channels)

# #     # Split into training and testing datasets
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         data, labels, test_size=0.2, random_state=42
# #     )

# #     # Initialize and train the SVM model
# #     model = SVC(kernel="linear", probability=True, random_state=42)
# #     print("Training the SVM model...")
# #     model.fit(X_train, y_train)

# #     # Evaluate the model
# #     y_pred = model.predict(X_test)
# #     accuracy = accuracy_score(y_test, y_pred)
# #     print(f"Model Accuracy: {accuracy * 100:.2f}%")

# #     # Get the unique labels in y_test
# #     unique_labels = np.unique(y_test)

# #     # Ensure target_names only contain class labels for the classes in y_test
# #     target_names = [class_labels.get(label, str(label)) for label in unique_labels]

# #     print("\nClassification Report:")
# #     print(classification_report(y_test, y_pred, target_names=target_names))

# #     # Save the trained model and class labels
# #     with open("traffic_sign_model.pkl", "wb") as f:
# #         pickle.dump((model, class_labels), f)
# #     print("Model saved as traffic_sign_model.pkl")


# # if __name__ == "__main__":
# #     train_model()
