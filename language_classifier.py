import nltk
from sentence_types import load_encoded_data
from sentence_types import encode_data, encode_phrases, import_embedding

from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

from keras.preprocessing.text import Tokenizer


class LanguageClassifier:
    # Use can load a different model if desired
        # Model configuration
    maxlen = 500
    batch_size = 64
    embedding_dims = 75
    filters = 100
    kernel_size = 5
    hidden_dims = 350
    epochs = 2

    # Add parts-of-speech to data
    pos_tags_flag = True



    def __init__(self,model_name = "models/cnn", embedding_name  = "data/default", load_model_flag = False):
        # load json and create model
        json_file = open(model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        
        # load weights into new model
        self.model.load_weights(model_name + ".h5")
        print("Loaded sentence classifier model from disk")
        
    
    def encode(comments):
        # Import prior mapping
        word_encoding, category_encoding = import_embedding()

        # Encode comments word + punc, using prior mapping or make new
        encoded_comments, word_encoding, \
            word_decoding = encode_phrases(comments, word_encoding,
                                           add_pos_tags_flag=True)
        
        

        padded_comments = sequence.pad_sequences(encoded_comments, maxlen=LanguageClassifier.maxlen)

        return padded_comments

    def predict(self, comments):
        encoded_comments = LanguageClassifier.encode(comments)

        
        annotations = []
        
        for c in comments:
            annotations.append(nltk.pos_tag(nltk.word_tokenize(c)))


        return annotations,self.model.predict(encoded_comments)
   


