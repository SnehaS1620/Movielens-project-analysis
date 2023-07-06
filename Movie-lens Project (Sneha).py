#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd

#pd.set_option('max_columns',None)
#pd.set_option('max_rows',None)

import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# In[11]:


from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import the three datasets

# In[12]:


movies = pd.read_csv('movies.dat' , delimiter = '::' , names = ['movieid' , 'title' , 'genres'],encoding='latin-1')

ratings = pd.read_csv('ratings.dat' , delimiter = '::' , names = ['userid' , 'movieid' , 'rating' , 'timestamp'],encoding='latin-1')

users = pd.read_csv('users.dat' , delimiter = '::' , names = ['userid' , 'gender' , 'age' ,'occupation' ,'zip_code'],encoding='latin-1')


# In[ ]:


#Users Data


# In[13]:


users.head()


# In[95]:


users.isnull().sum()


# In[14]:


users.shape


# In[ ]:


#Movies Data


# In[15]:


movies.head()


# In[16]:


movies.isnull().sum()


# In[17]:


movies.shape


# In[19]:


#ratings Data


# In[18]:


ratings.head()


# In[20]:


ratings.isnull().sum()


# In[21]:


ratings.shape


# ### Create a new dataset [Master_Data] with the following columns MovieID Title UserID Age Gender Occupation Rating.
# ### (Hint: (i) Merge two tables at a time.
# ### (ii) Merge the tables using two primary keys MovieID & UserId

# In[22]:


df1=pd.merge(users.drop('zip_code', axis=1), ratings.drop('timestamp',axis=1), on='userid')

master_data = pd.merge(df1, movies,   on='movieid')


# In[23]:


master_data.head()


# ### Explore the datasets using visual representations (graphs or tables), also include your comments on the following:

# In[ ]:


#Data Visualization


# ### 1. User Age Distribution

# In[25]:


age_count = users['age'].value_counts()
age_count


# In[26]:


age_Category = ('Under 18','18-24','25-34','35-44','45-49','50-55','56+')
x_position = np.arange(len(age_Category))
x_position


# In[27]:


age_Values =[age_count[1],age_count[18],age_count[25],age_count[35],age_count[45],age_count[50],age_count[56]]
age_Values


# In[30]:


#plotting bar chart
style.use('ggplot')
plt.figure(figsize=(9,8))
plt.bar(x_position,age_Values,align='center',color='r',alpha=0.7)
#set the y axis lable
plt.xlabel('Age Groups')
#set the bar value
plt.xticks(x_position,age_Category)
#set the title
plt.title('User Age Distribution')
plt.show()


# In[ ]:


# The above age distribution shows that most of people are from age group 25-34


# ### 2. User ratings of movie Toy Story

# In[31]:


movies.movieid[movies.title=='Toy Story (1995)']


# In[32]:


toystory_data = ratings[ratings.movieid==1]
toystory_data.head(10)


# In[33]:


movies_ratings_toystory = toystory_data.groupby('rating').size()
movies_ratings_toystory


# In[34]:


ratings_type = ('1','2','3','4','5')
x_pos = np.arange(len(ratings_type))
x_pos


# In[35]:


#plotting bar chart
style.use('ggplot')
plt.figure(figsize=(7,7))
plt.bar(x_pos,movies_ratings_toystory,align='center',color='r',alpha=0.7)
#set the y axis lable
plt.xlabel('ratings')
#set the bar value
plt.xticks(x_pos,ratings_type)
#set the title
plt.title('User ratings of movie Toy Story')
plt.show()


# In[ ]:


# The above group shows that the movie Toy story has ratings more than 4


# ### 3.Top 25 movies by viewership rating

# In[37]:


movies_rating = master_data.groupby(['title'], as_index=False)
average_movies_ratings = movies_rating.agg({'rating':'mean'})
average_movies_ratings.head(25)


# In[38]:


top_25_movies = average_movies_ratings.sort_values('rating', ascending=False).head(25)
top_25_movies


# In[39]:


top_25_movies.plot(kind='barh',alpha=0.6,figsize=(7,7))
plt.xlabel("Viewership Ratings Count")
plt.ylabel("Movies (Top 25)")
plt.title("Top 25 movies by viewership rating")
plt.show()


# ### 4. Ratings for all the movies reviewed by for a particular user of user id = 2696

# In[40]:


users_rating_data = master_data[master_data['userid']==2496]
users_rating_data = users_rating_data[['userid','movieid','title','rating']]
users_rating_data.head(10)


# In[41]:


# plotting the above data
plt.scatter(x=users_rating_data['movieid'].head(20),y=users_rating_data['rating'].head(20))
plt.show()


# ### Feature Engineering:
# ##### Use column genres:
# #### 1. Find out all the unique genres

# In[42]:


genres = master_data['genres'].str.split("|")
genres


# In[43]:


unique_genres = set()
for gen in genres:
    unique_genres = unique_genres.union(set(gen))


# In[44]:


unique_genres


# #### 2. Create a separate column for each genre category with a one-hot encoding ( 1 and 0) whether or not the movie belongs to that genre.

# In[46]:


onehotgenre = master_data["genres"].str.get_dummies("|")
onehotgenre.head()


# In[47]:


onehotgenre = pd.concat([master_data,onehotgenre],axis=1)
onehotgenre.head()


# In[48]:


onehotgenre.columns


# #### 3. Determine the features affecting the ratings of any particular movie

# In[49]:


features_data =master_data.copy()
features_data


# In[50]:


#Fetching the year which the movie was released
features_data[["title","year"]] = features_data.title.str.extract("(.)\s\((.\d+)",expand=True)
features_data = features_data.drop(['title'],axis=1)
features_data


# In[51]:


#Calculating the age of movies
features_data['year'] = features_data.year.astype(int)
features_data['movie_age'] = 2000 -features_data['year']
features_data


# In[52]:


#Creating Gender variable as integer type
features_data['gender'] = features_data.gender.replace('F',1)
features_data['gender'] = features_data.gender.replace('M',0)
features_data['gender'] = features_data.gender.astype(int)
features_data.head()


# In[53]:


#Checking the correlation of features with Rating
features_data[['gender','occupation', 'age', 'movie_age']].corrwith(features_data['rating'])


# In[54]:


#Occupation relationship with Rating
features_data.groupby(["occupation","rating"]).size().unstack().plot(kind='bar',stacked=False,legend=True)
plt.show()


# In[56]:


#Gender relationship with Rating
#1 -> Male, 0 -> Female


# In[57]:


features_data.groupby(["gender","rating"]).size().unstack().plot(kind='bar',stacked=False,legend=True)
plt.show()


# In[58]:


#Age relationship with Rating
features_data.groupby(["age","rating"]).size().unstack().plot(kind='bar',stacked=False,legend=True)
plt.show()


# In[59]:


#Movie_Age relationship with Rating
features_data.groupby(["movie_age","rating"]).size().unstack().plot(kind='bar',stacked=False,legend=True)
plt.show()


# #### 4. Develop an appropriate model to predict the movie rating

# In[60]:


#To Predict the values of rating we are using Logistic regression


# In[61]:


# Assign independent variables to X dataset
X = master_data[['age','occupation','movieid']].head(500)
X


# In[62]:


# Assign dependent variables to Y dataset
Y = master_data['rating'].head(500)
Y


# In[63]:


# view the shape for both axes
print (X.shape)
print (Y.shape)


# In[64]:


# Splitting the data into training & testing datasets(70:30)
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y,random_state=2,test_size=0.3)


# In[65]:


# use the Logistic regression estimator
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()


# In[67]:


# fit data into the Logistic regression estimator
logReg.fit(X_train,Y_train)


# In[68]:


y_predict=logReg.predict(X_test)


# In[69]:


y_predict


# In[70]:


# Calculate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_score(y_predict,Y_test)


# In[71]:


#Check model performance on new dataset
# create Example object with new values for prediction
X_new = [[25,7,1193],[18,17,2198]]


# In[72]:


logReg.predict(X_new)


# In[73]:


from sklearn import metrics
print (metrics.confusion_matrix(Y_test, y_predict))
print (metrics.classification_report(Y_test, y_predict))


# In[ ]:





# In[ ]:




