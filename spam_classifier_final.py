import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words


# this sec is to make a dictionary of the words in the email

vocabulary = {}
data = pd.read_csv("data/emails.csv")
set_words = set(map(str.strip, open('words.txt')))


def build_vocabulary(curr_email):
    idx = len(vocabulary)
    for word in curr_email:
        if word.lower() not in vocabulary and word.lower() in set_words:
            vocabulary[word] = idx
            idx += 1


if __name__ == "__main__":
    for i in range(data.shape[0]):
        curr_email = data.iloc[i, :][0].split()
        print(
            f"Current email is {i}/{data.shape[0]} and the \
               length of vocab is curr {len(vocabulary)}"
        )

        build_vocabulary(curr_email)

# Write dictionary to vocabulary.txt file
file = open("vocabulary.txt", "w")
file.write(str(vocabulary))
file.close()


file = open("vocabulary.txt", "r")
contents = file.read()
vocabulary = eval(contents)

X = np.zeros((data.shape[0], len(vocabulary)))
y = np.zeros((data.shape[0]))

for i in range(data.shape[0]):
    email = data.iloc[i, :][0].split()

    for email_word in email:
        if email_word.lower() in vocabulary:
            X[i, vocabulary[email_word]] += 1

    y[i] = data.iloc[i, :][1]

# Save stored numpy arrays
np.save("data/X.npy", X)
np.save("data/y.npy", y)


class NaiveBayes:
    def __init__(self, X, y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6

    def fit(self, X):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}

        for c in range(self.num_classes):
            X_c = X[y == c]

            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        probs = np.zeros((self.num_examples, self.num_classes))

        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.density_function(
                X, self.classes_mean[str(c)], self.classes_variance[str(c)]
            )
            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)

    def density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        # stripped from internet
        const = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.eps))
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs

# driver
if __name__ == "__main__":
    # run build_vocab to have .npy files
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")

    NB = NaiveBayes(X, y)
    NB.fit(X)
    y_pred = NB.predict(X)

    print(f"Accuracy: {sum(y_pred==y)/X.shape[0]}")
