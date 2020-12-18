from pandas import read_csv


def get_reviews_data(fpath):
    """Reads csv file, converts it to a pandas Datframe with columns 'text' and 'tag', and shuffles the dataframe
    :param fpath: csv file path
    :return: pandas DataFrame
    """

    data = read_csv(fpath)
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffles the dataset
    print(data.head())
    return data


def main():
    # Location of datasets
    folder = r'/Users/akshitagarwal/Desktop/FastAPI Krish Naik/Movie Reviews Sentiment Analysis/movie-reviews-sentiment-analysis/Datasets/'
    file = 'movie_reviews.csv'
    fpath = folder + file
    # fpath = r"/Users/akshitagarwal/Desktop/FastAPI Krish Naik/Movie Reviews Sentiment Analysis/movie-reviews-sentiment-analysis/Datasets/movie_reviews.csv"
    data = get_reviews_data(fpath)
    print(data.head())
    print(data.shape)
    print('test')

x
if __name__ == '__main__':
    main()
