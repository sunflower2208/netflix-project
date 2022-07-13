# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:14:31 2022

@author: Martyna
"""

import pandas as pd
import numpy as np

df = pd.read_pickle("input_data.pkl")


## Leave only the 1k top users to decrease computing time
users = df.userID.value_counts()[:1001].index.tolist()
df = df[df.userID.isin(users)]
df.userID.nunique()

df.genres = df.genres.fillna("None")
df.genres = df.genres.apply(lambda x: "None" if x == '\\N' else x)
df = df[df.genres != "None"] # drop rows with missing "genres" info

## Random 150k rows from the dataset to decrease computing time
df = df.sample(n=150000)
df.reset_index(inplace=True)
df.drop('index', inplace=True, axis = 1)

df['movie'] = df.type.apply(lambda x: 1 if 'movie' in x else 0) # content type encoding

genres = []
for i in range(len(df)):
    keywords = df.genres[i].split(",")
    for word in keywords:
        genres.append(word)
        
genres = set(genres) # list of unique categories in "genres" column

for genre in genres:
    df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0) # encoding of "genres"


################## MODEL BUILDING ##################

# Following rating prediction is a hybrid one. Rating is computed in 4 separate steps:
#     1. based on cosine similarity of users who rated the movie (computed with kNearestNeighbors)
#     2. based on the user's taste - how does the user rate similar movies - computed with linear regression
#     3. based on average mean rating of particular movie by all users in the training set - with no weighing
#     4. based on linear regression model, which takes into account all available attributes of the item: genres, release year etc.


## Dataset splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(["rating"], axis = 1), df['rating'], test_size = 0.33, random_state=1)
train = pd.concat([X_train, y_train], axis = 1)

X_train.to_pickle("X_train.pkl")
y_train.to_pickle("y_train.pkl")
X_test.to_pickle("X_test.pkl")
y_test.to_pickle("y_test.pkl")


################## 1. step of the hybrid system
from sklearn.neighbors import NearestNeighbors
user_movie_df = train.pivot(index='userID', columns='movieID', values='rating').fillna(0) # user-item matrix



def find_neighbors(userID, movieID, n_neighbors):
    """
    from users who rated movieID return k-most similar users to userID based on kNN
    """
    n_neighbors += 1
    train_movieIDs = train.movieID.unique()
    if movieID in train_movieIDs:
        rated = user_movie_df[user_movie_df[movieID] !=0]
        userIDs = train.userID.unique()
        users_ix = rated.index.tolist()
        users_ix.append(userID) # gives indices of users who rated movie and user from test set
        if userID in userIDs:
            data = user_movie_df.loc[users_ix, :]
            if len(data) < n_neighbors: n_neighbors=len(data)
            kNN = NearestNeighbors(metric='cosine', n_neighbors=n_neighbors, n_jobs=-1)
            kNN.fit(data)
            kNN_input = user_movie_df.loc[userID]
            out = kNN.kneighbors(kNN_input.array.reshape(1,-1))
            user_ix = out[1][0].tolist() # indices of neighbors in the 'data'
            user_sim = out[0][0].tolist() # cosine distance of the neighbors to userID
            sim_users = []
            data['userID'] = data.index
            for _ in range(1,len(user_ix)): # to exclude userID from the output
                sim_users.append([data.iloc[user_ix[_], -1], user_sim[_]])
            return sim_users
        else:
            print("userID not found in the training set")       
            return []
    else:
        print("movieID not in the training set")
        return []



def calc_rating(sim_users, movieID):
    """
    calculate rating of the movie based on similar users
    """
    if len(sim_users) > 1:
        weighted_rating_sum = 0
        cos_sim_sum = 0
        for i in sim_users:
            user_id = i[0]
            cos_sim_user = 1- i[1]
            movie_rating = train.loc[(train.movieID == movieID) & (train.userID == user_id),'rating'].values
            if len(movie_rating) > 0: movie_rating = movie_rating[0] # rating of movieID by a similar user
            weighted_rating_sum += (movie_rating * (cos_sim_user)) # gives sum of movieID rating multiplied by cosine similarity of particular user
            cos_sim_sum += cos_sim_user
        if cos_sim_sum == 0: return "NA"
        return weighted_rating_sum/cos_sim_sum # returns weighted average of movie rating
    elif len(sim_users) == 1:
        if sim_users[0][1] < 1:
            movie_rating = train.loc[(train.movieID == movieID) & (train.userID == sim_users[0][0]),'rating'].values
            if len(movie_rating) > 0: movie_rating = movie_rating[0]
            return movie_rating     
    else:
        return "NA"
    
    
    
X_test['similar_users'] = X_test.apply(lambda x: find_neighbors(x['userID'], x['movieID'], 10), axis = 1) # top 10 similar users with corresponding cosine distance
X_test['user_based_rating'] = X_test.apply(lambda x: calc_rating(x['similar_users'], x['movieID']), axis = 1) # returns weighted average rating of movieID


# to check accuracy of the step
# y_pred_1 = X_test['user_based_rating']
# y_pred_ub = y_pred_1[X_test['user_based_rating'] != "NA"]
# y_test_ub = y_test[X_test['user_based_rating'] != "NA"]

# from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
# mae = mean_absolute_error(y_test_ub, y_pred_ub) # out: 1.0771470894731257
# r2 = r2_score(y_test_ub, y_pred_ub) # out: -0.29177132839379927
# rmse = mean_squared_error(y_test_ub, y_pred_ub, squared=False) # out: 1.3378696589290118


################## 2nd step of the hybrid system
# which columns contain info about the movie?
movie_attributes = train.columns.tolist()
rem_attr = ["ratingDate", "type", "title", "keywords", "genres"]

for col in rem_attr:
    movie_attributes.remove(col)

movie_df = train[movie_attributes] # small DF with movie info
movie_attr_nrat = movie_attributes.copy()
movie_attr_nrat.remove("rating")

from sklearn import linear_model
        


def rate_movies(userID, test_row):
    """
    return predicted rating based on similar movies rated by the user
    """
    test_row = test_row[movie_attr_nrat]
    userIDs = movie_df.userID.unique() # which userIDs are in the training set?
    print(userID) # to track progress when run
    if userID in userIDs:
        rated = movie_df[movie_df['userID'] == userID]
        y = rated['rating']
        X = rated.drop("rating", axis = 1)
        lin_reg = linear_model.LinearRegression()
        lin_reg.fit(X,y)
        rating = lin_reg.predict(test_row.array.reshape(1,-1))
        if rating > 0:
            return rating
    else:
        print("userID not in the training set")
        return



X_test['taste_based'] = X_test.apply(lambda row: rate_movies(row['userID'], row), axis = 1) # returns predicted rating value of movie based on user preferences
X_test['taste_based'] = X_test['taste_based'].fillna(0).apply(lambda x: float(x) if x else x)

# to check accuracy of the step
# y_pred_2 = X_test['taste_based']
# y_pred_ub = y_pred_2[X_test['taste_based'] <= 5]
# y_test_ub = y_test[X_test['taste_based'] <= 5]

# mae = mean_absolute_error(y_test_ub, y_pred_ub) # out: 0.8677218230707034
# r2 = r2_score(y_test_ub, y_pred_ub) # out: 0.015573309347742259
# rmse = mean_squared_error(y_test_ub, y_pred_ub, squared=False) # out: 1.161877600288526



################## 3rd step of the hybrid system
avr_movie_rat = train.copy().drop('userID', axis = 1)
avr_movie_rat = avr_movie_rat.groupby("movieID").mean('rating') 

X_test['avr_movie_rating'] = X_test.movieID.apply(lambda x: avr_movie_rat.loc[x,'rating'] if x in avr_movie_rat.index else "NA") # returns average rating of movie

# to check accuracy of the step
# y_pred_3 = X_test['avr_movie_rating']
# y_pred_ub = y_pred_3[X_test['avr_movie_rating'] != 'NA']
# y_test_ub = y_test[X_test['avr_movie_rating'] != "NA"]

# mae = mean_absolute_error(y_test_ub, y_pred_ub) # out: 0.9259175309543913
# r2 = r2_score(y_test_ub, y_pred_ub) # out: 0.018991652370424328
# rmse = mean_squared_error(y_test_ub, y_pred_ub, squared=False) # out: 1.1610132665156272



################## 4th step of the hybrid system : when userID and movieID are not in training set
rating_df = train[movie_attributes]
rating_df.drop(["userID", "movieID"], inplace=True, axis = 1)
rating_df.drop_duplicates(inplace=True)
X = rating_df.drop("rating", axis = 1)
y = rating_df["rating"]

rating_model = linear_model.LinearRegression()
rating_model.fit(X,y)

X_test["lin_reg"] = rating_model.predict(X_test[X.columns.tolist()])

# to check accuracy of the step
# y_pred_4 = X_test["lin_reg"]

# mae = mean_absolute_error(y_test, y_pred_4) # out: 0.9419922580240842
# r2 = r2_score(y_test, y_pred_4) # out: -0.005767114983394217
# rmse = mean_squared_error(y_test, y_pred_4, squared=False) # out: 1.1822829636082688



################################# FINAL PREDICTION #################################

y_pred = X_test[['user_based_rating', 'taste_based', 'avr_movie_rating', 'lin_reg']]



def final_rating(row):
    """
    calculates average rating based on predictions from all columns
    ignores value if "NA", >5 or <= 0 value in cell
    """
    print(row[0])
    sum_rates = 0
    n_rates = 0
    for _ in range(len(row)):
        if type(row[_]) == float and row[_] > 0 and row[_] <= 5 and row[_] != "NA":
            sum_rates += row[_]
            n_rates += 1
    return round(sum_rates/n_rates)



y_pred['rating'] = y_pred.apply(lambda row: final_rating(row), axis = 1)


# mae = mean_absolute_error(y_test, y_pred['rating']) # out: 1.0016565656565657
# r2 = r2_score(y_test, y_pred['rating']) # out: -0.21329875525690878
# rmse = mean_squared_error(y_test, y_pred['rating'], squared=False) # out: 1.2965065726517906
