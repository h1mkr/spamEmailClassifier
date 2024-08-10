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

# Function to train perceptron with a specified number of epochs
def train_perceptron_with_epochs(train_data, weights, word_list_index, epochs):
    for epoch in range(epochs):
        check = True
        print('Epoch:', epoch)

        for index in range(len(train_data)):
            data = train_data[index]
            label = 1 if data['label'] == 'spam' else 0

            features = preprocess_text(data['text'], word_list_index)

            y = np.dot(weights, features) > 0  # Prediction using dot product

            if y != label:
                check = False
                weights += (label - y) * features  # Update weights only on misclassification

        if check:
            print('Converged')
            break

    return weights

# Function to train perceptron
def train_perceptron(train_data, weights, word_list_index, epochs=5):
    return train_perceptron_with_epochs(train_data, weights, word_list_index, epochs)

# Function to test perceptron
def test_perceptron(test_data, weights, word_list_index):
    correct = 0
    total = 0

    for data in test_data:
        features = preprocess_text(data['text'], word_list_index)
        label = 1 if data['label'] == 'spam' else 0

        y = np.dot(weights, features) > 0

        if y == label:
            correct += 1
        total += 1

    return correct / total

# Main code
train_data = dataset['train']
test_data = dataset['test']

word_set = set()

# Create word list and index mapping
for data in train_data:
    word_set.update(data['text'].lower().split())

word_list = list(word_set)
word_list_index = {word: index for index, word in enumerate(word_list)}

num_features = len(word_list)
weights = np.zeros(num_features)

accuracy_list = []
# Train perceptron with a specified number of epochs
for i in range(1, 11):
    epochs = i
    print('Training perceptron with', epochs, 'epochs')
    weights = train_perceptron(train_data, weights, word_list_index, epochs)

    # Test perceptron and calculate accuracy
    accuracy = test_perceptron(test_data, weights, word_list_index)
    accuracy_list.append(accuracy)
    print('Epochs:', epochs, 'Accuracy:', accuracy)

    weights = np.zeros(num_features)

plt.plot(range(1, 11), accuracy_list)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Perceptron Accuracy vs. Epochs')
plt.show()