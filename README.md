# US Netflix project
#### -- Status: Active
 
The purpose of this project can be divided into three main objectives:
1. To find trends, patterns and insights in content available on US Netflix
2. To visualize the data
3. To build model for predition of the movie/tv show rating by a user

## Datasets
Datasets used for the realization of this project were collected from various sources. In this README file the specific datasets will be refered to as:
* IMDB dataset - created by merging of data available from the IMDB website: https://datasets.imdbws.com/ and https://www.imdb.com/interfaces/
	sets used: title.basics.tsv.gz, title.ratings.tsv.gz, title.crew.tsv.gz and name.basics.tsv.gz
* Netflix dataset - "Netflix Movies and TV Shows" from Kaggle: https://www.kaggle.com/datasets/shivamb/netflix-shows
* prize dataset - "Netflix Prize data" from Kaggle: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data


## Methods and technologies
### Methods used in the project include:
* Statistics
* Data processing
* Data visualization
* Machine learning
* Predictive modeling

### Technologies:
* Python (v. 3.9)
* Jupyter
* Pandas
* Numpy
* Scikit-learn
* Surprise
* NLTK
* Json
* Seaborn
* Matplotlib

Additionally external data visualization with Tableau Public was performed. To see interactive dashboard go to: https://public.tableau.com/app/profile/sunflower2208/viz/Netflix-project/Dashboard1

For data visualization purposes wordclouds were constructed based on solution provided by KenJee: https://github.com/PlayingNumbers/ds_salary_proj/blob/master/data_eda.ipynb


## Detailed description of the project
### Part 1: Data cleaning, EDA, visualization
The Netflix dataset is available in three versions: from 2019, 2020 or 2021. The 2021 one is not complete (it does not contain info from the whole year), but it was used nonetheless as it contains more entries than the previous two. The structure of the datasets is similar, i.e. the same columns are present in all three, hence almost the same procedures can be applied.
As the IMDB provides datasets with content rating, this data was used to enrich the Netflix datasets.

#### imdb_ratings.ipynb
By merging the available IMDB sub-datasets a bigger one was created, containg columns such as:
* titleType - type of content (movie/tv show) - video games and tv episodes were droped from the set as they are not present in the Netflix dataset
* primaryTitle
* originalTitle
* startYear - for movies marking the year of release, for tv shows concerning only the first episode/series
* averageRating - created by IMDB users
* numVotes
* primaryName - name of the director

In the original files more content types were present (such as tvMovie or tvSpecial), but to be able to merge more data with the Netflix dataset these categories were converted into either "movie" or "tv show". Some unneeded columns (e.g. director's year or birth or death) were droped.
After checking for missing values the imdb.pkl file was created, containg attributes of more then 530k movies and almost 100k tv shows.

#### data_cleaning.ipynb
Before the Netflix datasets were merged with the IMDB one they were checked for missing values. As a result a mixup in three rows was found in the 2021 dataset, where duration was wrongly enlisted under "rating" column.
After it was fixed a merge with IMDB data was performed on the content title and year of release to limit creation of multiple values being connected to the content with the same ID. Still, duplicates were found and were further removed with certain conditions, i.e. director's names from both IMDB and Netflix datasets should match for the row not to be removed, etc. 
The datasets were saved to pickle files after successful merging and removal of incorrect rows for use in the EDA and data visualization.

#### EDA, data vis
As stated previously, the Netflix datasets are similar to one another, so the same attributes were investigated in each one. Including the IMDB dataset, for the data analysis following columns were available:
* show_id
* type - movie or tv show
* title
* director - in some rows this information was missing, however the rows were not dropped as they contained a lot of other info
* cast - list of actors
* country - of production/origin
* date_added - to Netflix
* release_year
* rating - describing the content suitability for certain audiences, e.g. G (general) or R (Restricted), not to be mistaken with IMDB ratings
* duration - for movies in minutes, for tv shows in number of seasons
* listed_in - Netflix categories (e.g. comedy, action)
* description
* averageRating - rating supplied by IMDB users in a scale 1-10, where the higher the better
* numVotes

As this project is still active, not all patterns or insights were investigated at this point. Nonetheless some can be shared now (for details see the EDA_20**.ipynb notebooks).
The performed EDA on the Netflix datasets made it possible to find that the majority of the available content is made from movies (almost 70%). When the data is analized with the countries as an attribute than this value can reach even 92% (when only India-based content is taken into account). Other insights can be found, such as the country-content dependencies. For example the countries which provide the most content to US Netflix include: United States, India, United Kingdom, Canada or France. When categories are investigated then it's obvious that international movies, dramas or commedies are the most populated ones.
From the available columns additional attributes were extracted which made it possible to find actors whom are cast in the most movies or tv shows. For example Samuel L. Jackson was casted in 21 US-based movies (when content from 2021 is taken into account).

Many more insights were found through EDA, for example distribution of the content duration throughout the release years or even if there is a dependecy between content type or its category on the IMDB rating.

The EDA of this project is not finished. As datasets from 3 separate years are available, different patterns are under investigation. For example change in the content will be investigated: how much is added by the Netflix each year/month/day? Is there a pattern, for example are more documentaries released in summer or winter months? These insights will be added to the repo when completed.

Basic interactive data visualization on the 2021 Netflix dataset is available on my Tableau Public profile. See:  https://public.tableau.com/app/profile/sunflower2208/viz/Netflix-project/Dashboard1


### Part 2: Model building
The second part of this project involved building of the rating prediction model of the movie/tv show by a certain user.
The downloaded Netflix prize dataset is composed of 4 txt files, which contain information about user ID, given rating and its date of a certain movie/tv show (movieID). The dataset can be enriched by movie titles (provided along with the txt files), as well as IMDB ratings and genres used in previous part of the project.
First the merging of IMDB dataset with movie titles was performed (see rating_model/movieID_imdb_merge.py) in a manner similar to the one previously described. The 'type' category was extracted from 'title'.
The output dataset was added to the one with users. Additionally keywords were extracted from movie titles by tokenizing and stemming with the use of NLTK library. At present these attributes are not yet used for prediction but will be at the later date.
Missing values were checked: around 12% of the data did not have the "genres" info - they were droped since the dataset is large (>1000k of rows). 7 movies were missing the year of release - due to low number of the movies these values were added either manually (based on Google search findings) or computed based on remaining data. The cleaned data was then used to check the distribution of ratings before model building.

To build the prediction engine two approaches were taken:
* As the SVD algorithm (see rating_model/model_svd.py) of the Surprise library is a great and well-known method for recommendation systems, it was applied to the part of Netflix prize dataset (to save computation time). Only three variables were used for the training, i.e. userID, movieID and the rating. The RMSE over 5 cross-validations was centered around 1.02, while MAE = 0.82
* The second approach (see rating_model/hybrid_rating_model.py) is a hybrid one, composed of 4 separate predictions:
	1. based on cosine similarity of users who rated the movie (computed with kNearestNeighbors)
	2. based on the user's taste - how does the user rate similar movies - computed with linear regression
	3. based on average mean rating of particular movie by all users in the training set - with no weighing
	4. based on linear regression model, which takes into account all available attributes of the item: genres, release year etc.

The hybrid model was trained and tested on a smaller dataset to save computation time. Additional tuning of each part is needed to improve the accuracy and speed performance of the prediction as in the present form it takes much longer to predict ratings in comparison to very fast SVD.

#### -- Status: Active
