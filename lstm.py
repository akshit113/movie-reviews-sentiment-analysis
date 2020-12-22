import nltk
from keras_preprocessing.sequence import pad_sequences
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pandas import read_csv, DataFrame, concat
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Embedding, Dropout, LSTM, Dense
from tensorflow.python.keras.models import Sequential

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def get_reviews_data(fpath):
    """Reads csv file, converts it to a pandas Dataframe with columns 'text' and 'tag', and shuffles the dataframe
    :param fpath: csv file path
    :return: pandas DataFrame
    """

    data = read_csv(fpath)
    # data = data.head(100)
    data.sample(frac=1).reset_index(drop=True, inplace=True)  # Shuffles the dataset
    return data


def remove_special_chars(df):
    import re
    reviews = df['text'].values.tolist()
    cleaned = []
    for review in reviews:
        clean_string = re.sub('\W+', ' ', review).strip(' ')
        cleaned.append(clean_string)
    cleaned_df = DataFrame(cleaned, columns=['text'])
    return cleaned_df


def lemmatize(df):
    reviews = df['text'].values.tolist()
    lmtzr = WordNetLemmatizer()
    sentences = []

    for review in reviews:
        words = []
        for word in word_tokenize(review):
            lm = lmtzr.lemmatize(word)
            if len(lm) > 1:
                words.append(lm)
        sentence = " ".join(words)
        sentences.append(sentence)
    cleaned_df = DataFrame(sentences, columns=['text'])
    return cleaned_df


def remove_stopwords(df):
    stop_words = stopwords.words('english')
    reviews = df['text'].values.tolist()
    removed = []
    cleaned_list = []
    for review in reviews:
        cleaned = []

        word_tokens = word_tokenize(review)
        for word in word_tokens:
            if word not in stop_words:
                cleaned.append(word)
        removed = " ".join(cleaned)
        cleaned_list.append(removed)
    cleaned_df = DataFrame(cleaned_list, columns=['text'])
    return cleaned_df


def create_baseline_model(num_words, input_len):
    """This function creates a LSTM Network with an Embedding Layer
    :param X: train set
    :param input_dim: Input to the Embedding Layer. This should be greater than or equal to the vocabulary size.
    :param output_dim: Dimension of the dense embedding.
    :return:Sequential Model
    """
    model = Sequential()
    # print(tokenizer.word_index)
    # model.add(Embedding(input_dim, output_dim))
    model.add(Embedding(input_dim=num_words, output_dim=512, input_length=input_len))
    model.add(LSTM(512, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.2))
    # model.add(Dense(64, activation='tanh'))
    # model.add(Dropout(0.2))
    # model.add(Dense(128, activation='tanh'))
    # model.add(Dropout(0.9))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])
    print(model.summary())
    return model


def get_words_counter(df):
    from collections import Counter
    words_list = []
    reviews = df.values.tolist()
    for review in reviews:
        words = review[0].split(' ')
        words_list.extend(words)
    counts = Counter(words_list)
    return counts


def pad_inputs(x_train, x_test, counter, max_len):
    from keras_preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=len(counter))
    tokenizer.fit_on_texts(x_train)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    print(train_sequences[1:10])

    x_train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
    x_test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
    return x_train_padded, x_test_padded


def main():
    # Location of datasets
    folder = r'/Users/akshitagarwal/Desktop/FastAPI Krish Naik/Movie Reviews Sentiment Analysis/movie-reviews-sentiment-analysis/Datasets/'
    file = 'movie_reviews.csv'
    fpath = folder + file
    df = get_reviews_data(fpath)
    print(df.head())
    print(df.shape)
    df['tag'].replace({'pos': 1, 'neg': 0}, inplace=True)

    lemmatized = lemmatize(df)
    rem_special = remove_special_chars(lemmatized)

    rem_stopwords = remove_stopwords(rem_special)
    df = concat([rem_stopwords, df['tag']], axis=1)

    counter = get_words_counter(rem_stopwords)

    max_len = 20
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['tag'], test_size=0.2, random_state=0)
    x_train_padded, x_test_padded = pad_inputs(x_train, x_test, counter, max_len)
    print(x_train_padded.shape)

    model = create_baseline_model(num_words=len(counter), input_len=20)
    model.fit(x_train_padded, y_train, epochs=10, batch_size=128, verbose=2, validation_data=(x_test_padded, y_test))
    print('model is fit...')
    print('test')


if __name__ == '__main__':
    main()
