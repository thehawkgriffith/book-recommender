import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
import pickle
import random
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def main():
    warnings.simplefilter(action='ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required = True, help = "Enter a book name to fetch recommendations on")
    args = vars(ap.parse_args())
    book = pd.read_csv('./BX-Books.csv', sep=';', error_bad_lines = False, encoding = 'latin-1', warn_bad_lines=False)
    book.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    user = pd.read_csv('./BX-Users.csv', sep=';', error_bad_lines = False, encoding = 'latin-1', warn_bad_lines=False)
    user.columns = ['userID', 'Location', 'Age']
    rating = pd.read_csv('./BX-Book-Ratings.csv', sep = ';', error_bad_lines = False, encoding = 'latin-1', warn_bad_lines=False)
    rating.columns = ['userID', 'ISBN', 'bookRating']
    print("\nPlease wait while the recommendations are fetched...")
    combine_book_rating = pd.merge(rating, book, on = 'ISBN')
    columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    combine_book_rating = combine_book_rating.dropna(axis=0, subset = ['bookTitle'])
    book_ratingCount = (combine_book_rating.
                        groupby(by = ['bookTitle'])
                        ['bookRating'].
                        count().
                        reset_index().
                        rename(columns = {'bookRating': 'totalRatingCount'})
                        [['bookTitle', 'totalRatingCount']]
                       )
    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how='left')
    pd.set_option('display.float_format', lambda x: '%.3f' %x)
    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    combined = rating_popular_book.merge(user, left_on = 'userID', right_on = 'userID', how = 'left')
    us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
    us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)
    us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
    usc_rating_pivot = us_canada_user_rating.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
    usc_rating_matrix = csr_matrix(usc_rating_pivot.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(usc_rating_matrix)
    temp = ''
    for name in list(usc_rating_pivot.items())[0][1].keys():
        if name.lower().find(str(args['name']).lower()) != -1:
            temp = name
    if temp == '':
        temp = random.choice(list(usc_rating_pivot.items())[0][1].keys())
    distances, indices = model_knn.kneighbors(usc_rating_pivot.loc[temp].values.reshape(1, -1), n_neighbors = 11)
    preds = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print("\nRecommendations for {}:\n".format(args['name']))
        else:
            print('{}: {}'.format(i, usc_rating_pivot.index[indices.flatten()[i]]))
            preds.append(usc_rating_pivot.index[indices.flatten()[i]])
    pickle.dump(preds, open('recs.obj', 'wb'))

if __name__ == "__main__":
    main()