from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv('topic_classification_cleaned_data.csv')
df = df.dropna(subset=['content'])
vectorizer = TfidfVectorizer(max_features=2000)

X = vectorizer.fit_transform(df['content']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=75, random_state=42).fit(X_train, y_train)

predicted_label = model.predict(X_test)
print(classification_report(y_test,predicted_label))