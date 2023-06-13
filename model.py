import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
import re
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from nltk.corpus import stopwords


class build_sarcasm_model:

    def __init__(self):
        pass

    def BuildModel(self):
        """## *Get the data*"""
        data_1 = pd.read_json("archive-2/Sarcasm_Headlines_Dataset.json", lines=True)
        data_2 = pd.read_json("archive-2/Sarcasm_Headlines_Dataset_v2.json", lines=True)
        data =  pd.concat([data_1, data_2])

        head_lines = self.CleanTokenize(data)
        self.max_length = 25

        self.tokenizer_obj = Tokenizer()
        self.tokenizer_obj.fit_on_texts(head_lines)
        self.sequences = self.tokenizer_obj.texts_to_sequences(head_lines)
        self.model = load_model('good model.h5')
        print("Sarcasm neural net model built")


    """## *Clean the data*"""

    def clean_text(self, text):
        text = text.lower()
        
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)
        text = " ".join(filter(lambda x:x[0]!='@', text.split()))
        emoji = re.compile("["
                            u"\U0001F600-\U0001FFFF"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        
        text = emoji.sub(r'', text)
        text = text.lower()
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)        
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text) 
        text = re.sub(r"\'ll", " will", text)  
        text = re.sub(r"\'ve", " have", text)  
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"don't", "do not", text)
        text = re.sub(r"did't", "did not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"couldn't", "could not", text)
        text = re.sub(r"have't", "have not", text)
        text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        return text



    def CleanTokenize(self, df):
        head_lines = list()
        lines = df["headline"].values.tolist()

        for line in lines:
            line = self.clean_text(line)
            # tokenize the text
            tokens = word_tokenize(line)
            # remove puntuations
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            # remove non alphabetic characters
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words("english"))
            # remove stop words
            words = [w for w in words if not w in stop_words]
            head_lines.append(words)
        return head_lines


    def predict_sarcasm(self, s):
        x_final = pd.DataFrame({"headline":[s]})
        test_lines = self.CleanTokenize(x_final)
        test_sequences = self.tokenizer_obj.texts_to_sequences(test_lines)
        test_review_pad = pad_sequences(test_sequences, maxlen=self.max_length, padding='post')
        pred = self.model.predict(test_review_pad)
        print(pred)
        pred*=100
        if pred[0][0]>=50: return "It's a sarcasm!" 
        else: return "It's not a sarcasm."

sarcasm_model = build_sarcasm_model()
sarcasm_model.BuildModel()

