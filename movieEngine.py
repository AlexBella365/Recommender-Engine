#import sqlite3
import numpy as np
import pandas as pd
import ast
import time
from scipy.sparse import coo_matrix, lil_matrix
import itertools
from math import sqrt
import pickle

#%%
class MovieRecommender: 
    
    def __init__(self,connector,**kwargs):
        self.__sqlConnector = connector
        self.__setAttributes(**kwargs)
        self.loadModel()  

    def __setAttributes(self, **kwargs):
        self.K = kwargs.get('K',20)
        self.lmbda = kwargs.get('lmbda',0.05)
        self.gamma = kwargs.get('gamma',0.05)
        self.Epoch_max = kwargs.get('Epoch_max',50)
        self.threshold = kwargs.get('threshold',1e-3)
        self.rating_min = kwargs.get('rating_min',4)
        self.mode = kwargs.get('mode','normal')
        
    def loadModel(self,filename = 'modelDump.mdl'):
        file = open(filename,'rb')
        data = pickle.load(file)
        self.A = data['matrixA']
        self.B = data['matrixB']
        self.Rlil = data['matrixR']
        self.listUsers = data['users']
        self.listMovies = data['movies']
        file.close()
        print('### Model loaded')
        
    def saveModel(self,filename = 'modelDump.mdl'):
        file = open(filename,'wb')
        toSave = {'matrixA':self.A,'matrixB':self.B,'matrixR':self.Rlil,
                   'users':self.listUsers,'movies':self.listMovies}
        pickle.dump(toSave,file)
        file.close()
        print('### Model saved as: '+filename)
        
    def addData(self,userId,movieId,rating=None):
        ######################################################
        #Can add code here to check the validity of the inputs
        ######################################################
        timestamp = int(time.time())
        values = (userId,movieId,rating,timestamp)
        
        sqlQuery = "REPLACE INTO Ratings VALUES (?,?,?,?)"  #Replace entry if it exists already
        self.__sqlConnector.execute(sqlQuery,values)
        self.__sqlConnector.commit() 
        
        self.__updateModel(userId,movieId,rating)
        self.__updateRecommendation(userId)
        self.saveModel()
        
        print('Succesfully added entry:')
        print('userId-> {:} | movieId-> {:} | rating-> {:}'.format(userId,movieId,rating))
        
        return True
          
    def bestRatedMoviesGlobally(self):
        top5globally = self.__sqlConnector.execute('''SELECT movieId, round(avg(rating),3)
                                FROM Ratings
                                GROUP BY movieId
                                HAVING count(rating)>500
                                ORDER BY avg(rating) DESC
                                LIMIT 5''').fetchall()
        purifiedList = [x[0] for x in top5globally]
        return purifiedList
    
    def predictInterest(self,userId):
        sqlQuery = "SELECT recommendation FROM Recommendations WHERE userID=?"
        rawresult = self.__sqlConnector.execute(sqlQuery,(userId,)).fetchone()
        if rawresult is None:
            print('This user is not yet in the database, returning community best rated movies')
            return self.bestRatedMoviesGlobally()
        else:
            print('Best 5 movies tailored for this user')
            result = ast.literal_eval(rawresult[0])
            bestmovies = [x[0] for x in result]
            return bestmovies
        
    def checkMovieScore(self,movieId):
        sqlQuery = "SELECT score, numberRatings FROM MovieScores WHERE movieId = ?;"
        (score,numberRatings) = self.__sqlConnector.execute(sqlQuery,(movieId,)).fetchone()
        print(('Movie {:} has been rated {:}/5 by {:} people').format(movieId,score,numberRatings))
   
    def __predict(self,A,B):
        return A.dot(B.T)  
    
    def __handleUserMovie(self,userId,movieId,K):
        correction_coeff = 20/K
        if userId not in self.listUsers:
            self.A = np.append(self.A,np.random.rand(1,K)*correction_coeff,axis = 0)
            self.listUsers.append(userId)
            # Add rows to Rlil
            temp = self.Rlil.toarray()
            temp = np.append(temp,np.zeros([1,temp.shape[1]]),axis=0)
            self.Rlil = lil_matrix(temp)
        if movieId not in self.listMovies:
            self.B = np.append(self.B,np.random.rand(1,K)*correction_coeff,axis = 0)
            self.listUsers.append(userId)
             # Add columns to Rlil
            temp = self.Rlil.toarray()
            temp = np.append(temp,np.zeros([temp.shape[0],1]),axis=1)
            self.Rlil = lil_matrix(temp)          
        
        return self.listUsers.index(userId) , self.listMovies.index(movieId)
        
    
    def __updateModel(self,userId,movieId,rating):
        print('*** Updating coefficients ***')
        RSE_list = []
        flag = True
        epoch = 0
        
        # Add entries to A, B, R if ncessary
        user, movie = self.__handleUserMovie(userId,movieId,self.K)  # and return indices
        
        if rating is None:
            rating = 0
        
        while flag and (epoch<self.Epoch_max):
            total_squared_error = 0
            e = rating - self.__predict(self.A[user,:],self.B[movie,:])           # Calculate error
            self.A[user,:] += self.gamma * ( e * self.B[movie,:] - self.lmbda * self.A[user,:])  # Update user feature matrix
            self.B[movie,:] += self.gamma * ( e * self.A[user,:] - self.lmbda * self.B[movie,:]) # Update movie feature matrix
            total_squared_error += e**2
            epoch += 1
            RSE = sqrt(total_squared_error)
            RSE_list.append(RSE)        
            
            if self.mode == 'debug':
                print('** Epoch {:02} | RSE: {:.3f}'.format(epoch,RSE))
            
            if epoch > 1 and abs(RSE-RSE_list[-2])<self.threshold:
                flag = False
        print('*** Updating complete ***')
        return RSE_list

    def __updateRecommendation(self,userId,forceCommit=True):
        user = self.listUsers.index(userId)      # Convert from userId to user postion in matrices
        Q = self.__predict(self.A[user,:],self.B)          # Calculate predicitons for this user
        recommendedMoviesIdx = Q.argsort()[::-1].tolist() # Get movie idx in descending order
        alreadyRated = self.Rlil.rows[user]            # List of movies already rated by this user
        top5idx = [m for m in recommendedMoviesIdx if m not in alreadyRated][:5] # Get the 5 best recommendations
        top5scores = Q[top5idx].round(3).tolist()   # the scores associated with these movies
        top5movies = [self.listMovies[m] for m in top5idx]   # Convert to actual movieId
        top5 = list(zip(top5movies,top5scores))         # Make a list of tuples
        
        # Update database
        value = (userId,str(top5))
        self.__sqlConnector.execute('''REPLACE INTO Recommendations (userId,recommendation,updated)
                        VALUES (?,?,1)''',value)
        if forceCommit:
            self.__sqlConnector.commit()


#%%
class ModelBuilder:
    
    def __init__(self,connector,initializeAtStart=False,**kwargs):
        self.__sqlConnector = connector
        self.__setAttributes(**kwargs)
        if initializeAtStart:
            self.computeModel()          
    
    def __setAttributes(self, **kwargs):
        self.K = kwargs.get('K',20)
        self.lmbda = kwargs.get('lmbda',0.05)
        self.gamma = kwargs.get('gamma',0.05)
        self.Epoch_max = kwargs.get('Epoch_max',50)
        self.threshold = kwargs.get('threshold',1e-3)
        self.rating_min = kwargs.get('rating_min',4)
        self.mode = kwargs.get('mode','normal')
    
    def computeModel(self,save=True,savefile='modelDump.mdl'):
        self.Rsparse, self.listUsers, self.listMovies  = self.__buildRatingMatrix()
        self.A,self.B,self.RMSE_list = self.__createModel()
        self.Rlil = self.Rsparse.tolil()
        if save:
            self.saveModel(savefile)
    
    def __buildRatingMatrix(self):
        print('***********************************')
        print('** Starting import, please wait')
        sqlQuery = '''SELECT userId, movieId, rating
                    FROM Ratings
                    WHERE userId in
                    (SELECT userId FROM Ratings
                    GROUP BY userId
                    HAVING count(rating)>{:})
                    ORDER BY userId'''.format(self.rating_min)
        raw = pd.read_sql(sqlQuery,self.__sqlConnector)
        R = raw.pivot_table(index='userId', columns='movieId', values='rating',aggfunc='max')
        Rsparse = coo_matrix(R.fillna(0))
        
        listUsers = raw.userId.unique().tolist()
        listUsers.sort()
        listMovies = raw.movieId.unique().tolist()
        listMovies.sort()
        
        print('** Import finished')
        print('***********************************')
        return Rsparse, listUsers, listMovies     

    def __predict(self,A,B):
        return A.dot(B.T)
    
    def __createModel(self):
        (nu,nm) = self.Rsparse.shape
        N = self.Rsparse.getnnz()
        print(("** Optimization with {:} movies and {:} users").format(nm,nu))
        print(("** Density {:.2}%").format(100*N/(nu*nm)))    
        
        # Initializatio with uniform random matrices 
        correction_coeff = 20/self.K   # So predict(A,B) statistically lies in range [0,5]
        A = np.random.rand(nu,self.K)*correction_coeff
        B = np.random.rand(nm,self.K)*correction_coeff
                          
        RMSE_list = []
        flag = True
        epoch = 0
        
        while flag and (epoch<self.Epoch_max):
            total_squared_error = 0
            for user, movie, rating in itertools.zip_longest(self.Rsparse.row, self.Rsparse.col, self.Rsparse.data):
                e = rating - self.__predict(A[user,:],B[movie,:])           # Calculate error
                A[user,:] += self.gamma * ( e * B[movie,:] - self.lmbda * A[user,:])  # Update user feature matrix
                B[movie,:] += self.gamma * ( e * A[user,:] - self.lmbda * B[movie,:]) # Update movie feature matrix
                total_squared_error += e**2
            
            epoch += 1
            RMSE = sqrt(total_squared_error/N)
            RMSE_list.append(RMSE)        
            
            if self.mode == 'debug':
                print('** Epoch {:02} | RMSE: {:.3f}'.format(epoch,RMSE))
            
            if epoch > 1 and abs(RMSE-RMSE_list[-2])<self.threshold:
                flag = False
        print('*** Optimization complete ***')
        print('------------------------------------------')
        return A,B,RMSE_list

        
    def saveModel(self,filename):
        file = open(filename,'wb')
        toSave = {'matrixA':self.A,'matrixB':self.B,'matrixR':self.Rlil,
                   'users':self.listUsers,'movies':self.listMovies}
        pickle.dump(toSave,file)
        file.close()
        print('### Model saved as: '+filename)

#%% Depreacted functions    
    
    ''''
    def __resetUpdateFlag(self,userId):
        sqlQuery = "UPDATE Recommendations SET updated = 0 WHERE userId = ?"
        self.__sqlConnector.execute(sqlQuery,(userId,))
        self.__sqlConnector.commit()
        
    def loadRecommendations(self):
        sqlQuery = "SELECT * FROM Recommendations;"
        self.recommendations = pd.read_sql(sqlQuery,self.sqlConnector)
        print('*** Recommendations loaded ***')
        
    def loadRatings(self):
        sqlQuery = "SELECT * FROM Ratings;"
        self.ratings = pd.read_sql(sqlQuery,self.sqlConnector)
        print('*** Ratings loaded ***')
        
    
    def cleanDataframe(self):
        self.df = self.df.drop(self.df.columns[3], axis=1)
        self.df.dropna(inplace=True)
        self.df.sort_values('userId',inplace=True)
        self.df.reset_index(drop=True,inplace=True)
        Nu = self.df.userId.unique().shape[0]
        Nm = self.df.movieId.unique().shape[0]
        print('*** Dataframe cleaned ***')
        print ('Number of users = ' + str(Nu) + ' | Number of movies = ' + str(Nm))
    
    def computeRatings(self):
        self.R = self.df.pivot_table(index='userId', columns='movieId',values='rating',
                           aggfunc='max')#.to_sparse()
        # Sparse matrix representation saves a lot of memory. Typical denisty: 2%
        print('*** Ratings Matrix computed ***')
    
    def getAverageRating(self):
        return self.R.mean(numeric_only=True)
    
    
    def computeDistances(self):
        self.movie_distances = pairwise_distances(R.T.fillna(0), metric='cosine')
        print('*** Distances computed ***')
    
    
    
    
    *************************
    The matrix movie_distances is not so big and can be kept in memory for now
    Later, we can implement a save method
    *************************
    def saveDistances(self):
        for movie in movie_distances
        self.movie_distances.to_sql('Distances', self.sqlConnector,
                                    if_exists='replace',
                                    chunksize = 10000,                  #Batch of 10,000
                                    dtype={'timestamp':'TIMESTAMP'})
        self.sqlConnector.commit()
     

    def retrieveSimilar(movieId,number=5):
      neighbors = self.movie_distances[movieId]
      best = neighbors.argsort().tolist()
      best.remove(movieId)
      return best[:number]    #Best match
      
    '''
