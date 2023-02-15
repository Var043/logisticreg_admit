# importing libraries
#independent: gre, gpa & rank dependent: admit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
 
#  importing dataset 

data_set=pd.read_csv("LR-1.csv")
# data_set.head()

# now we split the dataset into a training set and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_set[['gre','gpa','rank']],data_set['admit'],test_size=0.25,random_state=0)


# # now we do feature scaling because values are lie in diff ranges

# from sklearn.preprocessing import StandardScaler
# st_x=StandardScaler()
# x_train=st_x.fit_transform(x_train)
# x_test=st_x.fit_transform(x_test)
# # print(x_train[0:10,:])


# finally we are trianing our logistic regression model

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=0)
model.fit(x_train,y_train) # train the model on training set

# after training model its time to use it to do predictions on testing data.

y_pred=model.predict(x_test)


# confusion_matrix: use to check the model performance...

# from sklearn.metrics import confusion_matrix
# cm=confusion_matrix(y_test,y_pred)
# print("confusion matrix : \n",cm)

# getting accuracy
from sklearn.metrics import accuracy_score
print("Accuracy : ",accuracy_score(y_test,y_pred))


# saving the model a pickle file
# pickle.dump(LogisticRegression,open('LR_model.pkl','wb'))
with open('LR_model.pkl','wb') as f:
    pickle.dump(model,f)
# loading the model to disk
# pickle.dump(LogisticRegression,'LR_model.pkl','rb')
with open('LR_model.pkl','rb') as f:
    load_model=pickle.load(f)

# # visualing the performance of our model.
# from matplotlib.colors import ListedColormap
# x_set,y_set=x_test,y_test 
# x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,
#                             stop=x_set[:,0].max()+1,step=0.01),
#                   np.arange(start=x_set[:,1].min()-1,
#                             stop=x_set[:,1].max()+1,step=0.0))
# plt.contourf(x1,x2,model.predict(
#     np.array([x1.ravel(),x2.ravel()]).T).reshape(
#     x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))

# plt.xlim(x1.min(),x1.max())
# plt.ylim(x2.min(),x2.max())

# for i,j in enumerate(np.unique(y_set)):
#     plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
#                 c=ListedColormap(('red','green'))(i),label=j)

# plt.title('model (Test set) ')
# plt.xlabel(' Admit ')
# plt.ylabel(' gre/gpa/rank ')
# plt.legend()
# plt.show()
