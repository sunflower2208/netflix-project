# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:02:54 2022

@author: Martyna
"""
import pandas as pd

df = pd.read_pickle("input_data.pkl")

df = df.sample(n=250000)
df.reset_index(inplace=True)
df.drop(['index', 'genres', 'averageRating', 'keywords', 'title'],
        inplace=True,
        axis=1)

### Model with the Surprise library

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, GridSearchCV

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'movieID', 'rating']], reader=reader)

svd = SVD()

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)

## Output of the cross validation:
# {'test_rmse': array([1.02193482, 1.01939073, 1.020927  , 1.0240076 , 1.0192208 ]),
#  'test_mae': array([0.82097662, 0.81979958, 0.82096121, 0.82374305, 0.82100988]),
#  'fit_time': (7.809316635131836,
#   7.911633729934692,
#   8.645500659942627,
#   11.147096157073975,
#   11.075717687606812),
#  'test_time': (0.38034963607788086,
#   0.3793034553527832,
#   0.31107139587402344,
#   0.5706648826599121,
#   0.45357179641723633)}

## Tuning of the algorithm parameters

param_grid = {
    'n_epochs': [10, 20, 30],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.2, 0.4, 0.6]
}

gs = GridSearchCV(SVD, param_grid=param_grid, measures=['RMSE', 'MAE'], cv=5)

gs.fit(data)

gs.best_score['rmse']
## Output: 1.015134097599549

gs.best_score['mae']
## Output: 0.8168001041250028

gs.best_params['rmse']
## Output: {'n_epochs': 20, 'lr_all': 0.01, 'reg_all': 0.2}
results_df = pd.DataFrame.from_dict(gs.cv_results)

svd = SVD(n_epochs=20, lr_all=0.01, reg_all=0.2)
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)
## Output:
# {'test_rmse': array([1.01319716, 1.01574379, 1.01349017, 1.02069129, 1.01361071]),
#  'test_mae': array([0.81703475, 0.8175689 , 0.81613889, 0.82151067, 0.81608986]),
#  'fit_time': (8.022621631622314,
#   8.029301166534424,
#   8.249870777130127,
#   8.485263109207153,
#   11.94459319114685),
#  'test_time': (0.3802835941314697,
#   0.24999117851257324,
#   0.2501943111419678,
#   0.406048059463501,
#   0.5034983158111572)}
