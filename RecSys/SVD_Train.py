import tensorflow as tf
import pandas as pd
import numpy as np
import time
from collections import deque

tf.reset_default_graph()
np.random.seed(11) 

total_no_of_users = 138494
total_no_of_movies = 131263

batch_size = 1000 # Batch size
k = 13 # No. of latent factors
epochs = 1 # No. of epochs

def get_data(path):
    # Load data from given path using pandas.
    data = pd.read_csv(path)
    # Total no. of ratings
    rows = len(data)
    # Randomly shuffling the order of the ratings rows.
    data = data.iloc[np.random.permutation(rows)].reset_index(drop=True)
    # Declaring our test-train split percentage.
    split = int(rows * 0.8)
    train = data[0:split]
    test = data[split:].reset_index(drop=True)
    # For creating random batches of dataset.
    train_matrix = np.transpose(np.vstack(np.array([train[i] for i in ["userId", "movieId", "rating"]])))
    test_matrix = np.transpose(np.vstack(np.array([test[i] for i in ["userId", "movieId", "rating"]])))
    return train, test, train_matrix, test_matrix

def model(users, movies, no_of_users, no_of_movies, dim = 17):
    # Global bias for the dataset and since there is only one global bias no need for creating embeddings.
    global_bias = tf.get_variable("global_bias", shape=[], initializer=tf.zeros_initializer())
    # User and movie bias with random zero initialization.
    user_bias = tf.get_variable("user_bias", shape=[no_of_users], initializer=tf.zeros_initializer())
    movie_bias = tf.get_variable("movie_bias", shape=[no_of_movies], initializer=tf.zeros_initializer())
    # User bias and movie bias embeddings.
    user_bias_embd = tf.nn.embedding_lookup(user_bias, users, name="user_bias_embeddings")
    movie_bias_embd = tf.nn.embedding_lookup(movie_bias, movies, name="movie_bias_embeddings")
    # User and movie weights for creatings embedding vectors with intialization of stddev 0.01.
    user = tf.get_variable("user", shape=[no_of_users, dim], initializer=tf.truncated_normal_initializer(.01))
    movie = tf.get_variable("movie", shape=[no_of_movies, dim], initializer=tf.truncated_normal_initializer(.01))
    # User and movie embeddings with dim latent factors.
    user_embd = tf.nn.embedding_lookup(user, users, name="user_embeddings")
    movie_embd = tf.nn.embedding_lookup(movie, movies, name="movie_embeddings")
    # Getting result by doing dot product of user and movie embeddings and adding other biases.
    res = tf.reduce_sum(tf.multiply(user_embd, movie_embd), 1, name="dot_product")
    res = tf.add(res, user_bias_embd, name="add_user_bias")
    res = tf.add(res, movie_bias_embd, name="add_movie_bias")
    res = tf.add(res, global_bias, name="add_global_bias")
    tf.add_to_collection("output_", res)
    # Regularization part.
    reg = tf.add(tf.nn.l2_loss(user_embd), tf.nn.l2_loss(movie_embd), name="reguralization")
    return res, reg
    
def loss(res, reg, ratings, learning_rate=.01, regularization_rate=0.7):
    # For calculating mean squared error.
    mse = tf.reduce_mean(tf.squared_difference(res, ratings), name="mean_square_error") 
    # Declaring constant for the regularization part(lambda).
    penalty = tf.constant(regularization_rate, dtype=tf.float32, name="lambda")
    # Final cost function.
    cost = tf.add(mse, tf.multiply(penalty, reg), name="final_cost_function")
    # Initializing our optimizer to minimize cost function with given learning rate.
    opt = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
    return cost, opt

def get_random_batches(inp, b_size):
    # Get random set of indexes from 0 to length of the dataset.
    idxs = np.random.randint(0, inp.shape[0], b_size)
    output = inp[idxs, :]
    return [output[:, i] for i in range(inp.shape[1])]    

# Loading data from the csv file
data_train, data_test, train_matrix, test_matrix = get_data("C://Users//sorablaze_11//Desktop//RecSys//movielens-recommender-master//movielens-recommender-master//data//ml-ratings-100k-sample.csv")
samples_per_batch_train = len(data_train) // batch_size
samples_per_batch_test = len(data_test) // batch_size
print(len(data_train), ' ', len(data_test), ' ', samples_per_batch_train, ' ', samples_per_batch_test,' ', train_matrix.shape, ' ', test_matrix.shape)

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
movie_batch = tf.placeholder(tf.int32, shape=[None], name="id_movie")
rating_batch = tf.placeholder(tf.float32, shape=[None], name='ratings')

output, regularizer = model(user_batch, movie_batch, no_of_users=total_no_of_users, no_of_movies=total_no_of_movies, dim=k)
cost, opt = loss(output, regularizer, rating_batch, learning_rate=0.10, regularization_rate=0.05)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    writer.close()
    print("%s\t%s\t%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time", "Train Percentage", "Test Percentage"))
    errors = deque(maxlen=samples_per_batch_train)
    start = time.time()
    for i in range(epochs):
        total_count = 0
        for j in range(samples_per_batch_train):
            users, movies, ratings = get_random_batches(train_matrix, batch_size)
            _, pred_batch = sess.run([opt, output], feed_dict={user_batch: users, movie_batch: movies, rating_batch: ratings})
            pred_batch = np.clip(pred_batch, 1.0, 5.0)
            errors.append(np.power(pred_batch - ratings, 2))
            temp_count = np.abs(pred_batch - ratings)
            total_count += len(temp_count[np.where(temp_count <= 0.5)])        
        percentage_train = total_count / (samples_per_batch_train * batch_size)
        train_err = np.sqrt(np.mean(errors))
        test_err = np.array([])
        total_count = 0
        for j in range(samples_per_batch_test):
            users, movies, ratings = get_random_batches(test_matrix, batch_size)
            pred_batch = sess.run(output, feed_dict={user_batch: users, movie_batch: movies})
            pred_batch = np.clip(pred_batch, 1.0, 5.0)
            test_err = np.append(test_err, np.power(pred_batch - ratings, 2))
            temp_count = np.abs(pred_batch - ratings)
            total_count += len(temp_count[np.where(temp_count <= 0.5)])        
        percentage_test = total_count / (samples_per_batch_test * batch_size)
        end = time.time()
        print("%02d\t%.3f\t\t%.3f\t\t%.3f secs\t%.3f\t\t\t%.3f" % (i, train_err, np.sqrt(np.mean(test_err)), end - start, percentage_train, percentage_test))
        start = end
    saver.save(sess, './save/model')


imported_meta = tf.train.import_meta_graph('./save/model.meta')

def test():
    with tf.Session() as sess:
        sess.run(init_op)
        imported_meta.restore(sess, './save/model')
        test_err = np.array([])
        ans = 0.0
        for i in range(samples_per_batch_test):
            users, movies, ratings = get_random_batches(test_matrix, batch_size)
            pred_batch = sess.run(output, feed_dict={user_batch: users, movie_batch: movies})
            pred_batch = np.clip(pred_batch, 1.0, 5.0)
            # print("Pred\tActual")
            idx = np.random.randint(0, batch_size)
            # print("%.3f\t%.3f" % (pred_batch[idx], ratings[idx]))
            test_err = np.append(test_err, np.power(pred_batch - ratings, 2))
            # print(np.sqrt(np.mean(test_err)))
            ans += np.sqrt(np.mean(test_err))
        ans /= samples_per_batch_test
        print("Test Error :-\t%.3f" %ans)

test()