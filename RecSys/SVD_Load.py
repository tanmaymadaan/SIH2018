import tensorflow as tf
import pandas as pd
import numpy as np
import time
from flask import Flask, jsonify, request

tf.reset_default_graph()
np.random.seed(11) 

total_no_of_users = 138494
total_no_of_movies = 131263

batch_size = 1000 # Batch size
k = 13 # No. of latent factors
epochs = 5 # No. of epochs

all_movies = []

movie_to_id = {}
id_to_movie = {}

movies_rated_by_user = {}

movies = pd.read_csv("C://Users//sorablaze_11//Desktop//RecSys//movielens-recommender-master//movielens-recommender-master//data//ml-movies.csv")
# Load data from given path using pandas.
data = pd.read_csv("C://Users//sorablaze_11//Desktop//RecSys//movielens-recommender-master//movielens-recommender-master//data//ml-ratings-100k-sample.csv")
# Total no. of ratings
rows = len(data)
# Randomly shuffling the order of the ratings rows.
data = data.iloc[np.random.permutation(rows)].reset_index(drop=True)
# Declaring our test-train split percentage.
for _, row in movies.iterrows():
    id_to_movie[int(row[0])] = row[1]
    movie_to_id[row[1]] = int(row[0])
    all_movies.append(row[0])

for _ in range(1, total_no_of_users + 1):
    movies_rated_by_user[_] = []        

for _, row in data.iterrows():
    movies_rated_by_user[int(row[0])].append(int(row[1]))

print('Loading of data done.')

imported_meta = tf.train.import_meta_graph('./save/model.meta')
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)
imported_meta.restore(sess,  './save/model.ckpt')
output = tf.get_collection('output_')[0]

app = Flask(__name__)
@app.route('/api', methods=['POST'])

def getRecommendations():
    userId = request.get_json(force=True)
    userId = userId['id']
    movies = [_ for _ in all_movies if _ not in movies_rated_by_user[userId]]
    users = np.array([userId] * len(movies))
    movies = np.array(movies)
    pred_batch = sess.run(output, feed_dict={'id_user:0': users, 'id_movie:0': movies})
    print('length of predicted batch', len(pred_batch))
    pred_batch = np.clip(pred_batch, 1.0, 5.0)
    ratings = [[movies[_], pred_batch[_]] for _ in range(pred_batch.shape[0])]
    ratings = sorted(ratings, key=lambda x : x[1], reverse=True)
    ratings = ratings[:10]
    print('Movies to be recommender:')
    for _ in range(10):
        print(id_to_movie[ratings[_][0]], ' ', ratings[_][1])
    print(type(id_to_movie[ratings[0][0]]))
    optdict = []
    for _ in range(10):
        optdict.append({'movie_name' : id_to_movie[ratings[_][0]]})
    return jsonify(optdict)

if __name__ == '__main__':
    app.run(port=9000, debug=True)
