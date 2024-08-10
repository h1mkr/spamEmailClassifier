from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dataset = load_dataset("Deysi/spam-detection-dataset")

# Function to preprocess text data
def preprocess_text(text, word_list_index):
    text = text.lower().split()
    features = np.zeros(len(word_list_index))
    for word in text:
        if word in word_list_index:
            features[word_list_index[word]] = 1
    return features

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_logistic_regression(train_data, weights, word_list_index, learning_rate=0.1):
    epochs = 10         # Number of epochs
    for epoch in range(epochs):
        print('Epoch:', epoch)

        for data in train_data:
            label = 1 if data['label'] == 'spam' else 0
            features = preprocess_text(data['text'], word_list_index)

            y_pred = sigmoid(np.dot(weights, features))
            error = label - y_pred
            weights += learning_rate * error * features

    print('Training finished')
    return weights

def test_logistic_regression(test_data, weights, word_list_index):
    correct = 0
    total = 0

    for data in test_data:
        features = preprocess_text(data['text'], word_list_index)
        label = 1 if data['label'] == 'spam' else 0

        y_pred = sigmoid(np.dot(weights, features))
        predicted_label = 1 if y_pred >= 0.5 else 0

        if predicted_label == label:
            correct += 1
        total += 1

    return correct / total

train_data = dataset['train']
test_data = dataset['test']

word_set = set()

for data in train_data:
    word_set.update(data['text'].lower().split())

word_list = list(word_set)
word_list_index = {word: index for index, word in enumerate(word_list)}

num_features = len(word_list)
weights = np.zeros(num_features)

accuracy_list = []
max_accuracy = 0
learning_rate_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for learning_rate in learning_rate_list:
    weights = np.zeros(num_features)
    weights = train_logistic_regression(train_data, weights, word_list_index, learning_rate)
    accuracy = test_logistic_regression(test_data, weights, word_list_index)
    max_accuracy = max(max_accuracy, accuracy)
    accuracy_list.append(accuracy)

print('Max Accuracy:', max_accuracy)
plt.plot(learning_rate_list, accuracy_list)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.show()