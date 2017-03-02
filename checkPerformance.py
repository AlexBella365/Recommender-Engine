import pickle
import numpy as np

#%%
def predict(A,B):
    return A.dot(B.T)

#%%
def RMSE(R,I,P):
    E = (R-I.multiply(P)).power(2)
    N = E.nnz  #Non zeros elements, i.e. elements where we did make a prediciton error
    return np.sqrt(E.sum()/N), N, np.sqrt(E.max())
    
#%%
filename = 'modelDump.mdl'
file = open(filename,'rb')
data = pickle.load(file)
file.close()
print('### Model loaded')

RMSE_list = data['listRMSE']
E_list = data['listErrors']
L = len(RMSE_list)

## Plot
import matplotlib.pyplot as plt

plt.figure()

ax1 = plt.gca()
ax1.plot(range(L), [100*e for e in E_list], 'b')
ax1.set_ylabel('% of error', color = 'b')
#ax1.tick_params('y',color = 'b')

ax2 = ax1.twinx()
ax2.plot(range(L), RMSE_list,'g')
ax2.set_ylabel('RMSE', color = 'g')
#ax2.tick_params('y',color = 'g')

plt.title('CF with SGD - Learning curve')
plt.xlabel('Number of Epochs')
plt.grid()

plt.tight_layout()
plt.savefig('Error curve',dpi=200)

#%%
#
Umax, Mmax = 20000,10000
#
A = data['matrixA'][:Umax,:]
B = data['matrixB'][:Mmax,:]
R = data['matrixR'][:Umax,:Mmax]
listUsers = data['users'][:Umax]
listMovies = data['movies'][:Mmax]

(nu,nm) = R.shape
N = R.nnz



#%% Define the train/test data
I = (R>0)
Ifold = np.random.randint(4,size = [nu,nm])>0   # 75% fold 

Itrain = I.multiply(Ifold>0)
Itrain.eliminate_zeros()
Rtrain = Itrain.multiply(R)

Itest = I.multiply(Ifold==0)
Itest.eliminate_zeros()
Rtest = Itest.multiply(R)

#%% Compute the prediction matrices
Q = predict(A,B)
Rhat = (2*Q).round()/2.0

#%%
RMSE_all, N_all, emax_all = RMSE(R,I,Rhat)
RMSE_train, N_train, emax_train = RMSE(Rtrain,Itrain,Rhat)
RMSE_test, N_test, emax_test = RMSE(Rtest,Itest,Rhat)

#%%
print('----- Train set -----')
print('{:} bad predictions out of {:} ({:.2%} ) | Max error: {:} | RMSE: {:.3}'
      .format(N_train,Itrain.nnz,N_train/Itrain.nnz,emax_train,RMSE_train))

print('----- Test set -----')
print('{:} bad predictions out of {:} ({:.2%} ) | Max error: {:} | RMSE: {:.3}'
      .format(N_test,Itest.nnz,N_test/Itest.nnz,emax_test,RMSE_test))

print('----- Total -----')
print('{:} bad predictions out of {:} ({:.2%} ) | Max error: {:} | RMSE: {:.3}'
      .format(N_all,N,N_all/N,emax_all,RMSE_all))


#%%
import sqlite3
database = './data/RecommenderSystem.db'
sqlConnector = sqlite3.connect(database)

from movieEngine import ModelBuilder

database = './data/RecommenderSystem.db'
sqlConnector = sqlite3.connect(database)

model = ModelBuilder(sqlConnector,mode='debug',Epoch_max=100)
model.initialize()
