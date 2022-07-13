# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:48:40 2022

@author: Martyna
"""

import pandas as pd

file_1 = pd.read_csv("netflix-prize\combined_data_1.txt", sep=",", header=None, names=["userID", "rating", "ratingDate"])
file_2 = pd.read_csv("netflix-prize\combined_data_2.txt", sep=",", header=None, names=["userID", "rating", "ratingDate"])
file_3 = pd.read_csv("netflix-prize\combined_data_3.txt", sep=",", header=None, names=["userID", "rating", "ratingDate"])
file_4 = pd.read_csv("netflix-prize\combined_data_4.txt", sep=",", header=None, names=["userID", "rating", "ratingDate"])

file = pd.concat( [file_1, file_2, file_3, file_4],ignore_index=True)

file_ix = file[file['userID'].str.contains(":")].index

for i in range(len(file_ix)):
    if i != len(file_ix)-1:
        file.loc[file_ix[i]+1 : file_ix[i+1]-1,'movieID'] = file.loc[file_ix[i],"userID"][:-1]
    else:
        file.loc[file_ix[i]+1 :,'movieID'] = file.loc[file_ix[i],"userID"][:-1]
        
df = file.drop(file_ix)

df.reset_index(inplace=True)
df.to_pickle("rating_model\netflix_prize.pkl")