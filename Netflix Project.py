#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import All Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# ## OTT Movies Dataset
# 
# ##Let's look dataset

# In[4]:


# Loading dataset

OTT = pd.read_csv("D:\\ML project\\Netflix Movie EDA & Recommendation\\netflix_titles.csv")
OTT.head()


# ### Dataset
# 
# ##Looks like dataset contain details about movies & Tv shows such as title, director, cast, relased year, rating, duration, description etc.

# In[5]:


OTT.info()


# In[6]:


## here dataset contain details about both movies and shows so let's make two diffrent dataframes for this two.
OTT_movies = OTT[OTT['type'] == 'Movie']
OTT_shows = OTT[OTT['type'] == 'TV Show']


# ### Data Preprocessing

# In[9]:


OTT.isna().sum()


# In[10]:


OTT[OTT['date_added'].isna()]


# In[11]:


OTT = OTT[OTT['date_added'].notna()]


# In[12]:


OTT.head()


# In[13]:


OTT['rating'].unique()


# In[14]:


OTT[OTT['rating'].isna()]


# In[15]:


rating_replacement = {
    5889 : 'TV-14',
    6827 : 'TV-14',
    7312 : 'PG-13',
    7537 : 'TV-Y'
}

for id, rating in rating_replacement.items():
    OTT.iloc[id, 8] = rating


# In[16]:


OTT.info()


# In[17]:


OTT['year_added'] = OTT['date_added'].apply(lambda x : x.split(', ')[-1])
OTT['month_added'] = OTT['date_added'].apply(lambda x : x.split(' ')[0])
#OTT['year_added'].head()
#OTT['month_added'].head()


# In[18]:


OTT['season_count'] = OTT.apply(lambda x :str(x['duration']).split(" ")[0] if "Season" in str(x['duration']) else "", axis = 1)
OTT['duration'] = OTT.apply(lambda x : str(x['duration']).split(" ")[0] if "Season" not in str(x['duration']) else "", axis = 1)
OTT.head()


# In[19]:


OTT.dtypes


# In[20]:


# year_added convert to integer
OTT['year_added'] = pd.to_numeric(OTT['year_added'])

OTT['duration'] = pd.to_numeric(OTT['duration'], errors='coerce')

# type convert to category
#OTT['type'] = pd.Categorical(OTT['type'])


# In[21]:


OTT.dtypes


# In[22]:


type_count = OTT['type'].value_counts()
#type_count
labels = type_count.index
sizes = type_count.values

plt.figure(figsize=(4,4))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')

plt.title('Distribution of "Movie" and "TV show" in OTT Data')
plt.show()


# In[ ]:


###univariate analysis


# In[23]:


plt.figure(figsize=(12, 10))
sns.set(style = 'darkgrid')
ax = sns.countplot(x='rating', data=OTT, order=OTT['rating'].value_counts().index[0:15])


# In[24]:


country_count = OTT['country'].value_counts()[:11]

country_count.plot(kind='bar', figsize=(10,6))

plt.title('Top country with highest number ogf movies & shows')
plt.show()


# In[25]:


released_year_ott = OTT.loc[OTT['release_year'] > 2010].groupby(['release_year', 'type']).agg({'show_id': 'count'}).reset_index()
added_year_ott = OTT.loc[OTT['year_added'] > 2010].groupby(['year_added', 'type']).agg({'show_id': 'count'}).reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter( 
    x=released_year_ott.loc[released_year_ott['type'] == 'Movie']['release_year'], 
    y=released_year_ott.loc[released_year_ott['type'] == 'Movie']['show_id'],
    mode='lines+markers',
    name='Movie: Released Year',
    marker_color='green',
))

fig.add_trace(go.Scatter( 
    x=released_year_ott.loc[released_year_ott['type'] == 'TV Show']['release_year'], 
    y=released_year_ott.loc[released_year_ott['type'] == 'TV Show']['show_id'],
    mode='lines+markers',
    name='TV Show: Released Year',
    marker_color='darkgreen',
))

fig.add_trace(go.Scatter( 
    x=added_year_ott.loc[added_year_ott['type'] == 'Movie']['year_added'], 
    y=added_year_ott.loc[added_year_ott['type'] == 'Movie']['show_id'],
    mode='lines+markers',
    name='Movie: Year Added',
    marker_color='orange',
))

fig.add_trace(go.Scatter( 
    x=added_year_ott.loc[added_year_ott['type'] == 'TV Show']['year_added'], 
    y=added_year_ott.loc[added_year_ott['type'] == 'TV Show']['show_id'],
    mode='lines+markers',
    name='TV Show: Year Added',
    marker_color='darkorange',
))

fig.show()


# In[26]:


from scipy.stats import norm

sns.distplot(OTT.loc[OTT['release_year'] > 2000, 'release_year'], fit=norm, kde=False)


# In[27]:


plt.figure(figsize=(10, 6))
sns.kdeplot(data=OTT['duration'], shade=True)


plt.title("Movie duration in min")
plt.show()


# In[ ]:


###mutivariate analysis


# In[28]:


from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding


# In[29]:


def calculate_mlb(series):
    mlb = MultiLabelBinarizer()
    mlb_df = pd.DataFrame(mlb.fit_transform(series), columns=mlb.classes_, index=series.index)
    return mlb_df


# In[30]:


OTT['genre'] = OTT['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 
OTT['genre'].head()


# In[31]:


def top_genres(df, title='Top ones'):
    genres_df = calculate_mlb(OTT['genre'])
    tdata = genres_df.sum().sort_values(ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tdata.index,
        y=tdata.values,
    ))
    fig.update_xaxes(categoryorder='total descending')
    fig.update_layout(title=title)
    fig.show()


# In[32]:


top_genres(OTT, title='Top movies and shows Genres')


# In[33]:


genres_df = calculate_mlb(OTT['genre'])

corr = genres_df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(10, 7))
pl = sns.heatmap(corr, mask=mask, cmap= "coolwarm", vmax=.5, vmin=-.5, center=0, linewidths=.5,
                 cbar_kws={"shrink": 0.6})
plt.show()


# In[34]:


def check_genre_contains(genres):
    for genre in genres:
        if genre in top_movies_genres:
            return True
    return False


# In[35]:


OTT['principal_genre'] = OTT['genre'].apply(lambda genres: genres[0])
OTT['principal_genre'].head()


# In[36]:


top_movies_genres = [
    'International Movies',
    'Dramas',
    'Comedies',
    'Documentaries',
    'Action & Adventure',
]


# In[37]:


year_genre_df = OTT[(OTT['principal_genre'].isin(top_movies_genres)) & (OTT['year_added'] >= 2017)].groupby(['principal_genre', 'year_added']).agg({'title': 'count'})
year_genre_df = year_genre_df.reset_index()
year_genre_df.columns = ['principal_genre', 'year_added', 'count']

fig = px.sunburst(year_genre_df, path=['year_added', 'principal_genre'], values='count')
fig.show()


# In[ ]:


###recommendation system


# In[38]:


def clean_data(x):
    return str.lower(x.replace(" ", ""))


# In[39]:


filldna = OTT.fillna('')


# In[40]:


features=['title','director','cast','listed_in','description']
filldna = filldna[features]


# In[41]:


for feature in features:
    filldna[feature] = filldna[feature].apply(clean_data)
    
filldna.head(2)


# In[42]:


def create_tags(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']


# In[43]:


filldna['tags'] = filldna.apply(create_tags, axis=1)


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(filldna['tags'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[45]:


filldna=filldna.reset_index()
indices = pd.Series(filldna.index, index=filldna['title'])


# In[46]:


def get_recommendations_new(title, cosine_sim=cosine_sim):
    title=title.replace(' ','').lower()
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return OTT['title'].iloc[movie_indices]


# In[47]:


get_recommendations_new('Kota Factory', cosine_sim)

