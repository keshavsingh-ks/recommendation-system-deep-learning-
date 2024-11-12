#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/PrateekCoder/Recommendation-Systems/blob/main/Content_Based_Movie_Recommendation_System_Using_Binary_Feature_Matrix.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## YouTube 
# ###https://youtu.be/fsdjFdBbbpI

# ## Connect the Colab File with Google Drive

# In[25]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[4]:


#Import all the required packages
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# In[5]:


# Load the movies.csv file into a Pandas dataframe
movies = pd.read_csv('gdrive/My Drive/datasets/movielens-10m/movies.csv')


# In[6]:


movies


# A binary feature matrix for genres is used to represent the genre information of the movies in a numerical form that can be used as input for a recommendation system. The idea behind this is to represent each movie with a set of binary features that indicate whether the movie belongs to a certain genre or not. This allows the system to compare movies based on their genre similarity, rather than other attributes like director, cast, or release year.
# 
# The output of the binary feature matrix for genres will be a matrix where each row corresponds to a movie and each column corresponds to a genre. If a movie belongs to a certain genre, the corresponding entry in the matrix will be 1, otherwise it will be 0. For example, if we have 5 movies and 3 genres (Action, Drama, and Comedy), the binary feature matrix might look like this:
# 
# 
# 
# ```
# Action	Drama	Comedy
# Movie 1	1	0	0
# Movie 2	0	1	1
# Movie 3	1	1	0
# Movie 4	0	0	1
# Movie 5	1	1	1
# ```
# 
# 
# This is one of the most basic ways of implementing a content-based recommendation system. Another common way is to use a term frequency-inverse document frequency (TF-IDF) approach, where the genre information is represented as a weighted sum of the genre terms instead of binary features.
# 

# In[7]:


# Create a binary feature matrix for the genres
genre_matrix = pd.get_dummies(movies['genres'].str.split("|").apply(pd.Series).stack()).sum(level=0)


# In[8]:


genre_matrix


# In[12]:


# Compute the cosine similarity matrix
similarity = cosine_similarity(genre_matrix)
similarity


# In[13]:


# Function to get the recommended movies
def get_recommendations(title, top_n=5):
    # Find the index of the movie with the given title
    idx = movies[movies['title'] == title].index[0]
    
    # Get the cosine similarity scores for the movie
    similarity_scores = list(enumerate(similarity[idx]))
    
    # Sort the similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top_n movie indices
    movie_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    
    # Return the top_n most similar movies
    return movies['title'].iloc[movie_indices]


# In[14]:


# Ask the user for the movie name
title = input("Enter the title of your favorite movie: ")


# In[15]:


# Get the recommended movies
print("Top 5 similar movies:")
print(get_recommendations(title))


# In[ ]:




