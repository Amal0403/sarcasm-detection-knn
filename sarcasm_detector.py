import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

# Step 1: Load Twitter Dataset

data_file = "twitter_training.csv"

# Load CSV (no header)
df = pd.read_csv(data_file, header=None)
df.columns = ["tweet_id", "entity", "sentiment", "Sentence"]

print("Dataset loaded successfully! Shape:", df.shape)
print(df.head())

# Step 2: Convert sentiment to binary label

# Example mapping: negative/sarcastic → 1, others → 0
df["Label"] = df["sentiment"].apply(
    lambda x: 1 if str(x).lower() in ["negative", "sarcastic", "sarcasm"] else 0
)

df = df[["Sentence", "Label"]]

print("\n Data after preprocessing:")
print(df.head())


# Step 3: Extract features and labels

X = df["Sentence"].astype(str).tolist()
y = df["Label"]


# Step 4: Load SentenceTransformer model

print("\n Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 5: Encode dataset into embeddings

print("\nEncoding sentences into embeddings...")
X_vectors = model.encode(X, batch_size=32, show_progress_bar=True)


# Step 6: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Train KNN Classifier

print("\nTraining KNN Classifier...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)

# Step 8: Evaluate Model

y_pred = knn.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Interactive Prediction

print("\n🤖 Type any sentence to check if it's sarcastic (type 'exit' to quit)\n")

while True:
    user_input = input(">> ")
    if user_input.lower() == "exit":
        print("Exiting... Goodbye!")
        break

    user_vector = model.encode([user_input])
    prediction = knn.predict(user_vector)[0]

    if prediction == 1:
        print("This sentence is **Sarcastic / Negative**!\n")
    else:
        print("This sentence is **Not Sarcastic / Positive**.\n")
