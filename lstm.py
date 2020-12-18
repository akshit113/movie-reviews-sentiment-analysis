import nltk
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pandas import read_csv, DataFrame, concat

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def get_reviews_data(fpath):
    """Reads csv file, converts it to a pandas Datframe with columns 'text' and 'tag', and shuffles the dataframe
    :param fpath: csv file path
    :return: pandas DataFrame
    """

    data = read_csv(fpath)
    # data = data.samples(frac=1).reset_index(drop=True)  # Shuffles the dataset
    print(data.head())
    return data


def remove_special_chars(df):
    import re
    reviews = df['text'].values.tolist()
    cleaned = []
    for review in reviews:
        clean_string = re.sub('\W+', ' ', review).strip(' ')
        cleaned.append(clean_string)
    print(cleaned)
    cleaned_df = DataFrame(cleaned, columns=['text'])
    print(cleaned_df)
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
    print(cleaned_df)

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
    cleaned_df = DataFrame(cleaned_list)
    return cleaned_df


def preprocess_text_data(data, max_features, max_len):
    """This function prepares the dataset for applying ML Algorithm by converting the text data into numbers
    :param data:
    :param max_features:the maximum number of words to keep, based on word frequency. Only the most common words are kept
    :param max_len: maximum length of sequemces generated in text_to_sequences
    :return: numpy array of sequence of numbers (converted from textual data)
    """

    tokenizer = Tokenizer(num_words=max_features, split=' ')
    reviews_list = data['text']
    tokenizer.fit_on_texts(reviews_list)
    print(tokenizer.word_index)
    """This is the vocabulary size of the dataset. It is the number of unique words in our dataset. This should be the
    input of the embedding layer (input_dim) in the network."""
    X = tokenizer.texts_to_sequences(reviews_list)
    X = pad_sequences(X, maxlen=max_len)
    Y = data.loc[:, ['tag']].values
    return X, Y


def main():
    # Location of datasets
    folder = r'/Users/akshitagarwal/Desktop/FastAPI Krish Naik/Movie Reviews Sentiment Analysis/movie-reviews-sentiment-analysis/Datasets/'
    file = 'movie_reviews.csv'
    fpath = folder + file
    df = get_reviews_data(fpath)
    print(df.head())
    print(df.shape)
    print(df['tag'].value_counts())
    df['tag'].replace({'pos': 1, 'neg': 0}, inplace=True)
    df = df.head(100)

    rem_special = remove_special_chars(df)
    lemmatized = lemmatize(rem_special)
    rem_stopwords = remove_stopwords(lemmatized)
    df = concat([rem_stopwords, df['tag']], axis=1)
    X, Y = preprocess_text_data(df, max_features=1200, max_len=100)
    print(X.shape)
    print(Y.shape)
    print('test')

    # x_train, x_test, y_train, y_test = train_test_split(df['text'], df['tag'], test_size=0.3, random_state=0)
    # print(x_test.shape)
    # print(x_train.shape)

    print('test')


if __name__ == '__main__':
    main()
