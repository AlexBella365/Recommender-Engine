#import sqlite3
import numpy as np
import pandas as pd
import ast
import time
from scipy.sparse import coo_matrix, lil_matrix, eye
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
        self.lmbda = kwargs.get('lmbda',0.01)
        self.gamma = kwargs.get('gamma',0.02)
        self.Epoch_max = kwargs.get('Epoch_max',50)
        self.threshold = kwargs.get('threshold',1e-3)
        self.mode = kwargs.get('mode','normal')
        
    def loadModel(self,filename = 'modelDump.mdl'):
        file = open(filename,'rb')
        data = pickle.load(file)
        self.A = data['matrixA']
        self.B = data['matrixB']
        self.Rlil = data['matrixR']
        self.listUsers = data['users']
        self.listMovies = data['movies']
        self.RMSE_list = data['listRMSE']
        self.E_list = data['listErrors']
        file.close()
        print('### Model loaded')
        
    def saveModel(self,filename = 'modelDump.mdl'):
        file = open(filename,'wb')
        toSave = {'matrixA':self.A,'matrixB':self.B,'matrixR':self.Rlil,
                   'users':self.listUsers,'movies':self.listMovies,
                   'listRMSE':self.RMSE_list,'listErrors':self.E_list}
        pickle.dump(toSave,file)
        file.close()
        print('### Model saved as: '+filename)
        
    def addData(self,userId,movieId,rating=None,save=True):
        #Check the validity of the inputs
        if (not isinstance(userId,int) or (userId<0) 
        or not isinstance(movieId,int) or (movieId<0) 
        or not (2*rating == int(2*rating))            # Check that it is a round or half-round number
        or (rating<0) or (rating>5)):
                return False
        ######################################################
        timestamp = int(time.time())
        values = (userId,movieId,rating,timestamp)
        
        sqlQuery = "REPLACE INTO Ratings VALUES (?,?,?,?)"  # Add (or replace) entry
        self.__sqlConnector.execute(sqlQuery,values)
        self.__sqlConnector.commit() 
        
        self.__updateModel(userId,movieId,rating)
        self.updateRecommendation(userId)
        if save:
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
        correction_coeff = 15/K
        (nu,nm) = self.Rlil.shape
        
        if userId not in self.listUsers:
            self.A = np.append(self.A,np.random.rand(1,K)*correction_coeff,axis = 0)
            self.listUsers.append(userId)
            # Add rows to Rlil
            self.Rlil = lil_matrix(eye(nu+1,nu).dot(self.Rlil))            
            #temp = self.Rlil.toarray()
            #temp = np.append(temp,np.zeros([1,temp.shape[1]]),axis=0)
            #self.Rlil = lil_matrix(temp)
            
        if movieId not in self.listMovies:
            self.B = np.append(self.B,np.random.rand(1,K)*correction_coeff,axis = 0)
            self.listMovies.append(movieId)
             # Add columns to Rlil
            self.Rlil = lil_matrix(self.Rlil.dot(eye(nm,nm+1)))             
            #temp = self.Rlil.toarray()
            #temp = np.append(temp,np.zeros([temp.shape[0],1]),axis=1)
            #self.Rlil = lil_matrix(temp)          
        
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

    def updateRecommendation(self,userId,forceCommit=True):
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
            self.initialize()          
    
    def __setAttributes(self, **kwargs):
        self.K = kwargs.get('K',20)
        self.lmbda = kwargs.get('lmbda',0.01)
        self.gamma = kwargs.get('gamma',0.02)
        self.Epoch_max = kwargs.get('Epoch_max',50)
        self.threshold = kwargs.get('threshold',1e-3)
        self.mode = kwargs.get('mode','normal')
    
    def initialize(self,save=True,savefile='modelDump.mdl'):
        self.Rsparse, self.listUsers, self.listMovies  = self.__buildRatingMatrix()
        self.A,self.B,self.RMSE_list,self.E_list = self.__createModel()
        self.Rlil = self.Rsparse.tolil()
        if save:
            self.saveModel(savefile)
    
    def __buildRatingMatrix(self):
        print('***********************************')
        print('** Starting import, please wait')
        sqlQuery = 'SELECT userId, movieId, rating FROM Ratings'
        raw = pd.read_sql_query(sqlQuery,self.__sqlConnector)
        listUsers = raw.userId.unique().tolist()
        listUsers.sort()
        listMovies = raw.movieId.unique().tolist()
        listMovies.sort()
        
        data = raw['rating'].tolist()
        rows = raw.userId.astype('category', categories = listUsers).cat.codes
        cols = raw.movieId.astype('category', categories = listMovies).cat.codes
        Rsparse = coo_matrix((data, (rows, cols)), shape=(len(listUsers), len(listMovies)))      
        
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
        correction_coeff = 15/self.K   # So predict(A,B) statistically lies in range [0,5]
        A = np.random.rand(nu,self.K)*correction_coeff
        B = np.random.rand(nm,self.K)*correction_coeff
                          
        RMSE_list = []
        E_list = []
        flag = True
        epoch = 0
        
        while flag and (epoch<self.Epoch_max):
            total_squared_error_normalized = 0
            incorrect_ratings = 0
            for user, movie, rating in itertools.zip_longest(self.Rsparse.row, self.Rsparse.col, self.Rsparse.data):
                e = rating - self.__predict(A[user,:],B[movie,:])           # Calculate error
                A[user,:] += self.gamma * ( e * B[movie,:] - self.lmbda * A[user,:])  # Update user feature matrix
                B[movie,:] += self.gamma * ( e * A[user,:] - self.lmbda * B[movie,:]) # Update movie feature matrix
                
                total_squared_error_normalized += e**2/N
                if abs(e)>=0.5:
                     incorrect_ratings +=1                     
            
            epoch += 1          
            
            E_list.append(incorrect_ratings/N)
            
            RMSE = sqrt(total_squared_error_normalized)
            RMSE_list.append(RMSE)        
            
            if self.mode == 'debug':
                print('** Epoch {:02} | RMSE: {:.3f}'.format(epoch,RMSE))
            
            if epoch > 1 and abs(RMSE-RMSE_list[-2])<self.threshold:
                flag = False
        print('*** Optimization complete ***')
        print('------------------------------------------')
        return A,B,RMSE_list,E_list
        
    def saveModel(self,filename):
        file = open(filename,'wb')
        toSave = {'matrixA':self.A,'matrixB':self.B,'matrixR':self.Rlil,
                   'users':self.listUsers,'movies':self.listMovies,
                   'listRMSE':self.RMSE_list,'listErrors':self.E_list}
        pickle.dump(toSave,file)
        file.close()
        print('### Model saved as: '+filename)