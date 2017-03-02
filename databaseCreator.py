#%% Create the database
import sqlite3
database = './data/RecommenderSystem.db'
sqlConnector = sqlite3.connect(database)

#%%
import pandas as pd
movies_source = './data/ml-20m/movies.csv'
df_movies = pd.read_csv(movies_source)
df_movies.set_index('movieId')

#######################################################
try:
    sqlConnector.execute("DROP TABLE Movies")
except:
    pass
sqlConnector.execute('''CREATE TABLE Movies
                     ( movieId INTEGER PRIMARY KEY,
                     title TEXT,
                     genres TEXT)''')
sqlConnector.commit()

    # Populate it from our CSV data source
df_movies.to_sql('Movies', sqlConnector,
          if_exists= 'append',
          index = 'MovieId',
          chunksize = 10000,                  #Batch of 10,000
          dtype={'timestamp':'TIMESTAMP'})

#%%
import pandas as pd
ratings_source = './data/ml-20m/ratings.csv'
df_all = pd.read_csv(ratings_source)

# For testing, take a  lighter database
df = df_all.sample(n=int(1e6),random_state=0)  
Nu = df.userId.unique().shape[0]
Nm = df.movieId.unique().shape[0]
print ('Number of users = ' + str(Nu) + ' | Number of movies = ' + str(Nm))

#%% Create the Ratings table
try:
    sqlConnector.execute("DROP TABLE Ratings")
except:
    pass
sqlConnector.execute('''CREATE TABLE Ratings
                     (userId INTEGER,
                     movieId INTEGER,
                     rating REAL,
                     timestamp TIMESTAMP,
                     PRIMARY KEY (userId,movieId))''')
sqlConnector.commit()

    # Populate it from our CSV data source
df.to_sql('Ratings', sqlConnector, if_exists= 'append',chunksize = 10000,   #Batch of 10,000
          index = ('userId','movieId'),dtype={'timestamp':'TIMESTAMP'})

#%% Create the MovieScores table
try:
    sqlConnector.execute("DROP VIEW MovieScores")
except:
    pass
sqlConnector.execute('''CREATE VIEW MovieScores AS
                     SELECT movieId,
                     round(avg(rating),2) AS score,
                     count(rating) AS numberRatings
                     FROM ratings
                     GROUP BY movieId
                     ORDER BY movieId''')
sqlConnector.commit()

#%% Create the UsersScores table
try:
    sqlConnector.execute("DROP VIEW UsersScores")
except:
    pass
sqlConnector.execute('''CREATE VIEW  UsersScores AS
                     SELECT userId,
                     round(avg(rating),2) AS score,
                     count(rating) AS numberRatings
                     FROM ratings
                     GROUP BY userId
                     ORDER BY userId''')
sqlConnector.commit()

#%% Create the Recommendations table
from math import ceil

try:
    sqlConnector.execute("DROP TABLE Recommendations")
except:
    pass
sqlConnector.execute('''CREATE TABLE Recommendations
                     (userId INTEGER PRIMARY KEY,
                     recommendation TEXT DEFAULT '',
                     updated INTEGER DEFAULT 0)''')
sqlConnector.commit()

    # Populate it with the top 5 yet to be seen movies for each user
from movieEngine import ModelBuilder, MovieRecommender
model = ModelBuilder(sqlConnector,mode='debug')
model.computeModel()

rec = MovieRecommender(sqlConnector)
listUsers = rec.listUsers

count = 0
batch = 1
sizeBatch = int(1e4)
for user in listUsers:
    #value = (user,str(rec.predictInterest(user)))
    #sqlConnector.execute('''INSERT INTO Recommendations (userId,recommendation,updated)
    #                    VALUES (?,?,1)''',value)
    rec.updateRecommendation(user,forceCommit=False)
    count +=1 
    if count>=sizeBatch:
        print('#### Batch {:} of {:}'.format(batch,ceil(Nu/sizeBatch)))
        sqlConnector.commit()
        count = 0
        batch +=1
sqlConnector.commit()
print('====> Batch {:} of {:}'.format(batch,ceil(Nu/sizeBatch)))

#%% Make sure to save everything and close the connection
sqlConnector.commit()
sqlConnector.close()