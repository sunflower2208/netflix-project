# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:48:40 2022

@author: Martyna
"""

import pandas as pd
import json


## Import data

df_mt = pd.read_csv("movie_titles.csv", header=None, names=["movieID"])
mt_genres = pd.read_pickle("imdb_genres.pkl")



## Splitting the contents of 'movieID' column to get movie ID, its release year and the title

df_mt['movieID'] = df_mt['movieID'].apply(lambda x: list(x.split(";")))
df_mt["releaseYear"] = df_mt['movieID'].apply(lambda x: x[1])
df_mt['releaseYear'] = df_mt['releaseYear'].apply(lambda x: -999 if x == 'NULL' else int(x))
df_mt["title"] = df_mt['movieID'].apply(lambda x: x[2])
df_mt["movieID"] = df_mt['movieID'].apply(lambda x: x[0])

df_mt['type'] = df_mt["title"].apply(lambda x: "tv show" if ": series " in x.lower() or ": season " in x.lower() else "movie")
df_mt['title'] = df_mt['title'].apply(lambda x: x[:x.lower().find(": s")] if ": series " in x.lower() or ": season " in x.lower() else x)



#Add genres to the dataset based on IMDB data

mt_genres['startYear'] = mt_genres['startYear'].apply(lambda x: -999 if x == '\\N' else int(x))
mt_genres['to_drop'] = mt_genres['startYear'].apply(lambda x: 1 if int(x) > 2005 else 0)
mt_genres = mt_genres[mt_genres['to_drop'] != 1]
mt_genres.drop('to_drop', axis = 1, inplace = True)

df_mt_genres = pd.merge(df_mt, mt_genres, how = 'left',
                        left_on = ["title", 'type'],
                        right_on = ["primaryTitle", 'titleType'])
df_mt_genres.drop_duplicates(inplace=True, ignore_index = True)


duplicates = json.loads(df_mt_genres['movieID'].value_counts().to_json())
duplicate_keys = [int(k) for k,v in duplicates.items() if v > 1]


df_mt_genres['numVotes'] = df_mt_genres['numVotes'].fillna(0)
df_mt_genres['averageRating'] = df_mt_genres['averageRating'].fillna(0)
df_mt_genres['to_drop'] = df_mt_genres['averageRating'].apply(lambda x: 0)
df_mt_genres['to_keep'] = df_mt_genres['averageRating'].apply(lambda x: 0)
df_mt_genres['no_cat'] = df_mt_genres['genres'].apply(lambda x: len(str(x).split(",")))

df_mt_genres.sort_values(['movieID', 'startYear','numVotes','no_cat'], ignore_index=True, inplace = True, ascending=True)



## Get the indices of duplicates

indices = []

for i in range(len(df_mt_genres)):
    if int(df_mt_genres['movieID'][i]) in duplicate_keys:
        indices.append(i)
        
indices_check = indices.copy()


## Remove duplicates based on certain conditions
for i in range(len(indices)):
    c_movies = len(df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][indices[i]]) & (df_mt_genres['to_drop'] == 0)])
    count =  c_movies
    if df_mt_genres['releaseYear'][indices[i]] == df_mt_genres['startYear'][indices[i]]:
        df_mt_genres['to_keep'][indices[i]] = 1
        continue
    else:
        if (count > 1) and (df_mt_genres['releaseYear'][indices[i]] != 0):
            if df_mt_genres['startYear'][indices[i]] > df_mt_genres['releaseYear'][indices[i]]:
                df_mt_genres['to_drop'][indices[i]] = 1
                count -= 1
        if (count > 1) and len(df_mt_genres[['movieID', 'to_drop']][(df_mt_genres['movieID'] == df_mt_genres['movieID'][indices[i]])
                                                        & (df_mt_genres['to_drop'] == 0)]) > 1:
                if not df_mt_genres[['movieID', 'numVotes', 'to_drop']][(df_mt_genres['movieID'] == df_mt_genres['movieID'][indices[i]])
                                                             & (df_mt_genres['to_drop'] == 0)
                                                         & (df_mt_genres['numVotes'] > df_mt_genres['numVotes'][indices[i]])].empty:
                    df_mt_genres['to_drop'][indices[i]] = 1
    if df_mt_genres['to_drop'][indices[i]] == 1:
        indices_check.remove(indices[i])
    
df_mt_genres = df_mt_genres[df_mt_genres['to_drop'] != 1]


## Which movies have multiple rows?
movies_to_keep = []
movies_id = df_mt['movieID'].unique()

for i in movies_id:
    if len(df_mt_genres[df_mt_genres['movieID'] == i]) == 1:
        movies_to_keep.append(i)

indices_duplicates = []
        
for i in range(len(indices_check)): # Get indices of movieID duplicates
    if df_mt_genres['movieID'][indices_check[i]] not in movies_to_keep:
        indices_duplicates.append(indices_check[i])
        


## Remove duplicates based on conditions

for i in range(len(indices_duplicates)):
    ix = indices_duplicates[i]
    if df_mt_genres['to_drop'][ix] == 0 and df_mt_genres['to_keep'][ix] == 0:
        if len(df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][ix]) & (df_mt_genres['to_drop'] == 0) & (df_mt_genres['to_keep'] == 1)]) >= 1:
            df_mt_genres['to_drop'][ix] = 1
    if df_mt_genres['to_drop'][ix] == 0 and len(df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][ix]) & (df_mt_genres['to_drop'] == 0)]) > 1:
        if len(df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][ix]) & (df_mt_genres['to_keep'] == 1)]) == 1:
            if df_mt_genres['to_keep'][ix] == 0:
                df_mt_genres['to_drop'][ix] = 1
        elif len(df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][ix]) & (df_mt_genres['to_keep'] == 1)]) > 1 and df_mt_genres['to_keep'][ix] == 1:
            if not df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][ix]) & (df_mt_genres['to_drop'] == 0)
                             & (df_mt_genres['to_keep'] == 1) & (df_mt_genres['numVotes'] > df_mt_genres['numVotes'][ix])].empty:
                df_mt_genres['to_drop'][ix] = 1
        elif len(df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][ix]) & (df_mt_genres['to_keep'] == 1)]) == 0:
            if not df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][ix]) & (df_mt_genres['to_drop'] == 0)
                                                     & (df_mt_genres['numVotes'] > df_mt_genres['numVotes'][ix])].empty:
                df_mt_genres['to_drop'][ix] = 1
            elif not df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][ix]) & (df_mt_genres['to_drop'] == 0)
                               & ((df_mt_genres['releaseYear'] - df_mt_genres['startYear']) < (df_mt_genres['releaseYear'][ix] - df_mt_genres['startYear'][ix]))].empty:
                df_mt_genres['to_drop'][ix] = 1
        if df_mt_genres['to_keep'][ix] == 0 and df_mt_genres['to_drop'][ix] == 0:
            if len(df_mt_genres[(df_mt_genres['movieID'] == df_mt_genres['movieID'][ix]) & (df_mt_genres['to_drop'] == 0) & (df_mt_genres['to_keep'] == 0)]) > 1:
                df_mt_genres['to_drop'][ix] = 1
                             
df_mt_genres = df_mt_genres[df_mt_genres['to_drop'] != 1]


## Check for duplicates
    
duplicates_check = json.loads(df_mt_genres['movieID'].value_counts().to_json())
duplicate_keys_check = [int(k) for k,v in duplicates_check.items() if v > 1]

print(df_mt_genres[df_mt_genres['movieID'] == '11714'])

df_mt_genres.drop(4137, axis=0, inplace = True)
df_mt_genres.reset_index(inplace = True)

print(df_mt_genres.columns)

## Remove redundant columns

df_mt_genres.drop(labels = ['index', 'titleType', 'primaryTitle', 'startYear', 'numVotes', 'to_drop', 'to_keep', 'no_cat'], axis = 1, inplace = True)


# Save DataFrame to file

df_mt_genres.to_pickle('data_genres.pkl')
