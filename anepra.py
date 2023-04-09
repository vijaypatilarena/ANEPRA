import tensorflow as tf
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')


# Loading the dataset
dataset = ["The quick brown fox jumps over the lazy dog",
           "The brown fox jumps over the lazy dog",
           "The quick brown fox jumps over the lazy cat"]

# Define the stop words to remove from the text helps in clearing chaos caused by relentive excess data.
stop_words = set(stopwords.words('english'))

# Define the vectorizer to convert text to numeric vectors (For training data set and having it vectorized so it would remeber at next iterations)
vectorizer = TfidfVectorizer(stop_words=stop_words)

# Convert the dataset to numeric vectors (for application and iteration)
X = vectorizer.fit_transform(dataset)

# Defining the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(X, np.array([1, 0, 1]), epochs=100)

# Define a function to detect plagiarism
def detect_plagiarism(text1, text2):
    # Convert the text to numeric vectors
    vec1 = vectorizer.transform([text1])
    vec2 = vectorizer.transform([text2])
    # Calculate the cosine similarity between the vectors (Cosine simalirty will result in plagared elements from the internet.)
    similarity = np.dot(vec1.toarray(), vec2.toarray().T) / (np.linalg.norm(vec1.toarray()) * np.linalg.norm(vec2.toarray()))
    # Use the neural network to predict if the text is plagiarized
    prediction = model.predict(np.array([similarity]))
    if prediction > 0.5:
        return "Plagiarism detected!"
    else:
        return "No plagiarism detected."

# Test the plagiarism detector (Using the universal prompt which includes every other alpahbet from english language)
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "The brown fox jumps over the lazy dog"
print(detect_plagiarism(text1, text2))

text1 = "The quick brown fox jumps over the lazy dog"
text2 = "The quick brown cat jumps over the lazy dog"
print(detect_plagiarism(text1, text2))

#THIS PROJECT IS STILL UNDER DEVELOPMENT THIS IS A TRAINING PROTOTYPE#
