#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[5]:


movies = pd.read_csv('tmdb_movies_data.csv')


# In[6]:


movies.head()


# In[8]:


movies.describe()


# In[14]:


movies = movies[['id','original_title','overview','genres']]


# In[15]:


movies['tags']=movies['overview']+movies['genres']


# In[16]:


movies


# In[18]:


new_data = movies.drop(columns=['overview','genres'])


# In[19]:


new_data


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer


# In[22]:


cv=CountVectorizer(max_features=1000, stop_words='english')


# In[24]:


vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()


# In[26]:


vector.shape


# In[28]:


from sklearn.metrics.pairwise import cosine_similarity


# In[29]:


similarity=cosine_similarity(vector)


# In[30]:


similarity


# In[32]:


new_data[new_data['original_title']=="The Godfather"].index[0]


# In[41]:


distance=sorted(list(enumerate(similarity[2])),reverse=True, key=lambda vector:vector[1])
for i in distance[0:5]:
    print(new_data.iloc[i[0]].original_title)


# In[43]:


def recommand(movies):
    index=new_data[new_data['original_title']==movies].index[0]
    distance=sorted(list(enumerate(similarity[index])),reverse=True, key=lambda vector:vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].original_title)


# In[46]:


recommand("Batman")


# In[54]:


import pickle


# In[55]:


pickle.dump(new_data,open('movies_list.pkl','wb'))


# In[58]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[59]:


pickle.load(open('movies_list.pkl','rb'))


# In[ ]:




