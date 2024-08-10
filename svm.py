from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def predict_from_folder(folder_path, svm_classifier, vectorizer):
    predictions = []
    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), 'r') as f:
            text = f.read().lower()
            vector = vectorizer.transform([text])
            prediction = svm_classifier.predict(vector)
            print('File:', file, 'Prediction:', prediction[0])
            predictions.append(prediction[0])
    return predictions

# Load dataset
dataset = load_dataset("Deysi/spam-detection-dataset")

# Extract texts and labels
train_dataset = dataset['train']
test_dataset = dataset['test']

train_texts = [data['text'] for data in train_dataset]
train_labels = [1 if data['label'] == 'spam' else 0 for data in train_dataset]

test_texts = [data['text'] for data in test_dataset]
test_labels = [1 if data['label'] == 'spam' else 0 for data in test_dataset]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Vectorize training and testing texts
train_vectors = vectorizer.fit_transform(train_texts)
test_vectors = vectorizer.transform(test_texts)

# Regularization parameters to test
regularization_params = [0.1, 1, 2, 2.5, 3, 10, 100]

# Lists to store accuracies for each regularization parameter
accuracies_linear = []
accuracies_rbf = []

max_linear_accuracy = 0
max_rbf_accuracy = 0

# Loop through regularization parameters
for param in regularization_params:
    # Initialize SVM classifier with current regularization parameter
    classifier_linear = SVC(kernel='linear', C=param)
    classifier_rbf = SVC(kernel='rbf', C=param)

    # Train the classifier
    classifier_linear.fit(train_vectors, train_labels)
    classifier_rbf.fit(train_vectors, train_labels)

    # Test the classifier
    predictions_linear = classifier_linear.predict(test_vectors)
    predictions_rbf = classifier_rbf.predict(test_vectors)

    # Calculate accuracy and append to list
    accuracy = accuracy_score(test_labels, predictions_linear)
    accuracies_linear.append(accuracy)
    max_linear_accuracy = max(max_linear_accuracy, accuracy)
    accuracy = accuracy_score(test_labels, predictions_rbf)
    accuracies_rbf.append(accuracy)
    max_rbf_accuracy = max(max_rbf_accuracy, accuracy)

# Print maximum accuracies
print('Max Linear Accuracy:', max_linear_accuracy)
print('Max RBF Accuracy:', max_rbf_accuracy)

# Plot accuracy versus regularization type graph
plt.plot(regularization_params, accuracies_linear, marker='o')
plt.plot(regularization_params, accuracies_rbf, marker='x')
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Regularization Parameter')
plt.xscale('log')
plt.legend(['Linear Kernel', 'RBF Kernel'])
plt.grid(True)
plt.show()

classifier_linear = SVC(kernel='linear', C=2)
classifier_rbf = SVC(kernel='rbf', C=100)

classifier_linear.fit(train_vectors, train_labels)
classifier_rbf.fit(train_vectors, train_labels)

predictions_linear = predict_from_folder('./test', classifier_linear, vectorizer)
predictions_rbf = predict_from_folder('./test', classifier_rbf, vectorizer)