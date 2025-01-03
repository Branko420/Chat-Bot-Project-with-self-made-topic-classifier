import torch
import joblib
from topic_classifier_class import TopicClassifier
from cleanning_the_raw_data import preprocess_text

vectorizer = joblib.load("./tfidf_vectorizer.pkl")
label_encoder = joblib.load("./label_encoder.pkl")

num_classes = len(label_encoder.classes_)
feature_size = len(vectorizer.get_feature_names_out())

model = TopicClassifier(features_size=feature_size, num_classes=num_classes)
model.load_state_dict(torch.load("topic_classifier_state.pth"))
model.eval()

def topic_classify(prompt):
    preproduced_sentence = preprocess_text(prompt)
    input_features = vectorizer.transform([preproduced_sentence]).toarray()

    input_data = torch.tensor(input_features, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_data)
        predicted_class = torch.argmax(output,dim=1).item()

    predicted_topic = label_encoder.inverse_transform([predicted_class])
    return predicted_topic[0]