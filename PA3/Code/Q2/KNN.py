import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

# Load digit dataset
digits = load_digits()
# Get features and labels
X, y = digits.data, digits.target

# Split the dataset into training and testing sets (50 images per class for testing)
def train_test_custom_split(X, y, num_classes=10, num_samples_per_class=50):
    test_indices = []   # Initialize an empty list to store indices of test samples
    
    for class_label in range(num_classes):
        class_indices = np.where(y == class_label)[0]                                             # Get all indices of the current class label
        selected_indices = np.random.choice(class_indices, num_samples_per_class, replace=False)  # Randomly select a fixed number of samples for the test set from the current class
        test_indices.extend(selected_indices)                                                     # Add the selected indices to the test indices list

    test_indices = np.array(test_indices)       # Convert test indices list to a NumPy array
    train_indices = np.array([i for i in range(len(y)) if i not in test_indices])    # Create training indices by excluding test indices from the entire dataset
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_custom_split(X, y)

# k-Nearest Neighbor Classifier
def k_nearest_neighbor_classifier(X_train, y_train, X_test, k):
    y_pred = []     # Initialize an empty list to store predictions
    
    for test_point in X_test:
        distances = [distance.euclidean(test_point, x_train) for x_train in X_train]    # Calculate Euclidean distances between the test point and all training points
        nearest_indices = np.argsort(distances)[:k]                                     # Get the indices of the k nearest training points
        nearest_labels = [y_train[i] for i in nearest_indices]                          # Get the labels of the k nearest training points
        predicted_label = max(set(nearest_labels), key=nearest_labels.count)            # Determine the most frequent label among the k nearest neighbors
        y_pred.append(predicted_label)                                                  # Append the predicted label to the predictions
        
    return np.array(y_pred)

# Test the k-NN classifier for k=3, 5, and 7
for k in [3, 5, 7]:
    # Make predictions using the nearest neighbor classifier
    y_pred = k_nearest_neighbor_classifier(X_train, y_train, X_test, k)
    # Calculate accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification accuracy for k={k}: {accuracy * 100:.2f}%")
    print('_'*50)