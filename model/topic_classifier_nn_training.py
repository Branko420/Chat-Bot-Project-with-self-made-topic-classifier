import joblib
import torch
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from topic_classifier_class import TopicClassifier


df = pd.read_csv('topic_classification_cleaned_data.csv')
df = df.dropna(subset=['content'])
vectorizer = TfidfVectorizer(max_features=2000)

X = vectorizer.fit_transform(df['content']).toarray()
label_encolder = LabelEncoder()
y = label_encolder.fit_transform(df['label'])

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encolder, "label_encoder.pkl")

X_train_tensor = torch.tensor(X, dtype=torch.float32)
y_train_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_train_tensor, y_train_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

feature_size = X.shape[1]
num_classes = len(label_encolder.classes_) 

model = TopicClassifier(features_size=feature_size, num_classes=num_classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(14):
    model.train()
    for batch in loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch: {epoch+1}, loss: {loss.item():.4f}")
torch.save(model.state_dict(), 'topic_classifier_state.pth')
