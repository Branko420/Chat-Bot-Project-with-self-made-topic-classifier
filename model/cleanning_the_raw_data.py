import pandas as pd 
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df = pd.read_csv('./topic_classification_data.csv')

def preprocess_text(content):
    stop_words = set(stopwords.words('english'))
    content = content.lower()
    content = re.sub(r'<.*?>', '', content)
    content = re.sub(r'[^\w\s,]', '', content, flags=re.UNICODE)
    content = re.sub(r'\W+',' ', content)
    content = re.sub(r'[^a-zA-Z\s]', '', content)
    words = word_tokenize(content)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


df['content'] = df['content'].fillna("")
df['content']=df['content'].apply(preprocess_text)

df.to_csv('topic_classification_cleaned_data.csv', index=False)