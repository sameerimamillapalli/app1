#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot
import plotly.graph_objects as go
#import plotly.offline as iplot
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output# Load Data
#from jupyter_dash import JupyterDash
#from jupyter_dash import JupyterDash
import plotly.express as px
init_notebook_mode(connected=True)
cf.go_offline()
pd.options.display.float_format = "{:.2f}".format


# Modules:
# Numpy, Pandas {data analysis}, matplotlib,Seaborn  :{ Basic-Plotting,Themes},
# Plotly:{Visual-Plotting},sklearn:{Simple Imputer},cufflinks:{Reading pandas data to plotly and required themes},
# dash:{dash_core_components,dash_hrml_components},Jupyter-Das:{To work on dash in jupyter notebook}

# In[2]:


cf.getThemes()


# In[3]:


cf.set_config_file(theme="solar")


# In[4]:


#reading csv file 
df_country = pd.read_csv("country-wise-average.csv")


# In[5]:


df_country.head()#dataframe of first five rows


# In[6]:


pd.set_option('display.max_rows',152)#To print maximum rows assigned by the user


# In[7]:


df_country


# # Data-Preprocessing

# In[8]:


df_country.head(5)


# In[9]:


df_country.shape#shape of the dataset


# In[10]:


df_country.columns# columns of the datset 


# In[11]:


df_country.dtypes# datatypes in the dataset


# In[12]:


df_country.isnull().sum()#checking if the there is Nan in the data


# In[13]:


df_country.describe()#Descriptive Stats before Imputation


# # Imputation

# In[14]:


df_country= df_country.dropna(subset = ['Wasting', 'Overweight', 'Stunting', 'Underweight'])



df_country['Severe Wasting'].fillna(df_country['Severe Wasting'].mean(), inplace=True)


#Imputation Process for Severe Wasting with more null values


# In[15]:


df_country.isnull().sum()#checking null values after imputation


# In[16]:


df_country.info()


# In[17]:


df_country.describe()#Descriptive Stats after Imputation


# **Observation**:There is an increase in 0.2 approx across median and mean in the data .

# # Style Tabulation in Country Estimates 

# In[18]:


df_country['Income Classification'].value_counts()


# **Observation**:There are more countries Classified in Upper-Middle-Income in the Datast

# In[19]:


Country = df_country.groupby(['Income Classification'])


# In[20]:


Country.get_group(0).style.highlight_max(color='#26BB2D',subset=["Severe Wasting","Wasting","Overweight","Stunting","Underweight","U5 Population ('000s)"]).highlight_min(color='#E65236',subset=["Severe Wasting","Wasting","Overweight","Stunting","Underweight","U5 Population ('000s)"])


# **Observation**:
# 
# Severe Wasting%:  Max : South Sudan   Min: Burundi
# 
# Wasting%:         Max : South Sudan   Min: Uganda
# 
# Overweight%:      Max : Syrian Republic Min:Nepal
# 
# Stunting%:        Max :  Burundi     Min:Gambai 
# 
# Underweight%:      Max : NiGER       Min: Syrian Republic   
# 
# 1)South Sudan has both high Severe Wasting% as well Wasting%
# 
# 2)Syrian Republic has more overweight as well least underweight % in contrast
# 
# 3)Burundi has least Severe Wasting% , but on other hand has highest Stunting% which is not suggested
# 
# 
# 

# In[21]:


Country.get_group(1).style.highlight_max(color='#26BB2D',subset=["Severe Wasting","Wasting","Overweight","Stunting","Underweight","U5 Population ('000s)"]).highlight_min(color='#E65236',subset=["Severe Wasting","Wasting","Overweight","Stunting","Underweight","U5 Population ('000s)"])


# **Observation**:
# 
# 1)Bangladesh has more Undeweight% and least overweight% whcih clearly says that Income in Bangladesh for Health is under red zone
# 
# 2)REPUBLIC OF MOLDOVA has both least , Stunting% and Underweight% clearly depicts that their malnutrition status is improving.
# 
# 3)India has high Wasting% as well U5 population across lower middle Income

# In[22]:


Country.get_group(2).style.highlight_max(color='#26BB2D',subset=["Severe Wasting","Wasting","Overweight","Stunting","Underweight","U5 Population ('000s)"]).highlight_min(color='#E65236',subset=["Severe Wasting","Wasting","Overweight","Stunting","Underweight","U5 Population ('000s)"])


# **Observation**:
# 
# 1)Nauru has least severe Wasting% and as well U5 population% which is expected rate for not-Malnourished

# In[23]:


Country.get_group(3).style.highlight_max(color='#26BB2D',subset=["Severe Wasting","Wasting","Overweight","Stunting","Underweight","U5 Population ('000s)"]).highlight_min(color='#E65236',subset=["Severe Wasting","Wasting","Overweight","Stunting","Underweight","U5 Population ('000s)"])


# **Observation**:
# 
# 1)Australia has least parmater% except overweight% which is high across High Income . If decreased over Years might lead Australia first country in the world to succeed Malnutrition.

# # Visualization with Seaborn 

# In[24]:


plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
cols = ["Severe Wasting","Wasting","Overweight","Stunting","Underweight"]
sns.pairplot(df_country[cols], height = 2.5,corner=True )
plt.show();


# **Observation**:
# 
# 1)Stunting and Underweight has a Linear relationship 
# 
# 2)Wasting and Underweight also exhibits Linear Relationship
# 

# In[25]:


plt.figure(figsize=(12, 8))
x = df_country.groupby(["Income Classification"])["Severe Wasting"].mean()
sns.set(style="whitegrid")
ax = sns.barplot(x.index, x)
ax.set_title('Severe Wasting')
ax.set_ylabel('% Severe Wasting')
ax.set_xlabel('Income Classification')
plt.xticks(rotation = 90)


# **Observation**:
#     
# 1) Low-Income has High Severe Wasting% of 3-4% approx across all income Classes following up are Lower-middle-2.5%, upper-middle 1.5%,High-Income 1-2%

# In[26]:


plt.figure(figsize=(12, 8))
x = df_country.groupby(["Income Classification"])["Wasting"].mean()
sns.set(style="whitegrid")
ax = sns.barplot(x.index, x)
ax.set_title('Wasting')
ax.set_ylabel('% Wasting')
ax.set_xlabel('Income Classification')
plt.xticks(rotation = 90)


# **Observation**:
#     
# 1) Low-Income has High  Wasting% 10%,across all income Classes following up are Lower-middle 8%, upper-middle4-5%,High-Income3% approx

# In[27]:


plt.figure(figsize=(12, 8))
x = df_country.groupby(["Income Classification"])["Stunting"].mean()
sns.set(style="whitegrid")
ax = sns.barplot(x.index, x)
ax.set_title('Stunting')
ax.set_ylabel('% Stunting')
ax.set_xlabel('Income Classification')
plt.xticks(rotation = 90)


# **Observation**:
#     
# 1) Low-Income has High  Stunting% of 40% across all income Classes following up are Lower-middle(30-35)%, upper-middle(15-20)%,High-Income(5-10)%

# In[28]:


plt.figure(figsize=(12, 8))
x = df_country.groupby(["Income Classification"])["Overweight"].mean()
sns.set(style="whitegrid")
ax = sns.barplot(x.index, x)
ax.set_title('Overweight')
ax.set_ylabel('% Overweight')
ax.set_xlabel('Income Classification')
plt.xticks(rotation = 90)


# **Observation**:
#     
# Upper-Middle-Income has highest Overweight % around 8-10% approx following up are High-Income with 7% and 6-8% Lower-Middle-Income of 7%, low income 4-5% approx

# In[29]:


plt.figure(figsize=(12, 8))
x = df_country.groupby(["Income Classification"])["Underweight"].mean()
sns.set(style="whitegrid")
ax = sns.barplot(x.index, x)
ax.set_title('Underweight')
ax.set_ylabel('% Underweight')
ax.set_xlabel('Income Classification')
plt.xticks(rotation = 90)


#  **Observation**:
#     
# Low-Income has High  Stunting% of 20-25% across all income Classes following up are Lower-middle(15-20)%, upper-middle(5-10)%,High-Income(0-5)%
# 
# 
#  **Final Observation**:
#  
# 1)Intrestingly , Low-Income has high Parameter % across all plots which depicts that Income is main Effect for Malnutrition levels . They have insufficient food left and live with striped Muscles
# 
# 2)Overweight % is high in Upper-Middle-Income, High-Income as their income is high and as well they tend to have over-Nutrition as expected to other Incomes in terms of Quality and Saturated Fats

# In[30]:


fig=df_country.groupby(['Income Classification'])[['Severe Wasting','Wasting','Overweight','Stunting','Underweight']].mean().iplot(kind='bar', asFigure=True)
fig.show()


#  **Observation**:
#  
#  Across All Incomes Stunting% and Underweight% is dominated  more and Income wise Stunting% and Underweight% is decreasing .
#  Wasting , Severe Wasting, Overweight have a variability across different Incomes

# #  Box-Plots

# In[31]:


def income_map(val):#Mapping 0,1,2,3 Income to respective names low,low middle, upper middle ,high income
    mapper = {0:'Low Income', 1:'Lower Middle Income', 2:'Upper Middle Income',3:'High Income'}
    return mapper[val]


# In[32]:


df_country['Income level'] =df_country['Income Classification'].apply(income_map)


# In[33]:



fig2 = px.box(df_country, x='Income level', y='Severe Wasting',title='Severe Wasting among different income',color='Income level', 
             color_discrete_sequence=["green", "blue", "goldenrod", "magenta"])

fig2.show()


# In[34]:



fig3 = px.box(df_country, x='Income level', y='Wasting',title='Wasting among different income',color='Income level', 
             color_discrete_sequence=["green", "blue", "goldenrod", "magenta"])

fig3.show()


# **Observation**:
# 
# Low income,Lower Middle Income countries tend to have a higher level of wasting. As we all know, wasting is caused by numerous factors and one of them is low energy intake. People in these countries tend to survive on less than 3 meals and hence suffer from severe malnutrition. Their bodies switched to survival mode and their muscles are striped off to preserve energy.

# In[35]:


fig4 = px.box(df_country, x='Income level', y='Overweight',title='Overweight among different income',color=df_country['Income level'], 
             color_discrete_sequence=["green", "blue", "goldenrod", "magenta"])

fig4.show()


# **Observation**:
# 
# Unsurprisingly, we see high level of overweight issues in high income and upper middle income countries. As more people are able to afford their meals in these countries, we will tend to see a greater level of overweight issues.

# In[36]:


fig5 = px.box(df_country, x='Income level', y='Stunting',title='Stunting among different income',color=df_country['Income level'], 
             color_discrete_sequence=["green", "blue", "goldenrod", "magenta"])

fig5.show()


# **Observation**:
#     
# Both low and lower middle income countries have high median stunting rates.Upper-Middle-Income also exhibits Medium Stunting% . 
# Skewness in left of Lower-Middle-Income suggest that they are more from 10-30% countries Malnutritioned proned

# In[37]:




fig6 = px.box(df_country, x=df_country['Income level'], y=df_country['Underweight'],title='Underweight among different income',color=df_country['Income level'], 
             color_discrete_sequence=["green", "blue", "goldenrod", "magenta"])

fig6.show()


# **Observation**:
#     
# Both low and lower middle income have high rates of underweight issues.Lowe-Middle-Income Exhibits more Skewness on right which depicts an increase from 25-45% where food Take is insufficient across Lower-Middle-Income.
# Upper-Middle-Income Has Outliers More in number which suggests that some countries have more Underweight% than Expected.  

# In[ ]:





# In[38]:


#Plotting on the WorldMap using plotly
x1= df_country.groupby(["Country"])["Severe Wasting"].mean()


col_map1 = go.Figure(
                data=go.Choropleth(
                locations = x1.index,
                locationmode = 'country names',
                colorscale= 'portland',
                text= x1.index,
                z=x1,
                colorbar = {'title':'Severe Wasting % ', 'len':200,'lenmode':'pixels' },
                
            ), layout = go.Layout(geo=dict(bgcolor= 'rgba(255,255,255,0)',showframe=False, projection = {'type': 'natural earth'}),
                                  title = 'Severe Wasting % (South-Sudan)',
                                  
                                  titlefont = {"size": 15, "color":"White"},
                                  geo_scope='world',
                                  margin={"r":0,"t":40,"l":0,"b":0},
                                  paper_bgcolor='#000000',
                                 
                                  )
            )

col_map1.show()



# In[39]:


#Plotting on the WorldMap using plotly
x2 = df_country.groupby(["Country"])["Wasting"].mean()
col_map2 = go.Figure(
                data=go.Choropleth(
                locations = x2.index,
                locationmode = 'country names',
                colorscale= 'portland',
                text= x2.index,
                z=x2,
                colorbar = {'title':' Wasting %', 'len':200,'lenmode':'pixels' },
                
            ), layout = go.Layout(geo=dict(bgcolor= 'rgba(255,255,255,0)',showframe=False, projection = {'type': 'natural earth'}),
                                  title = ' Wasting % (South-Sudan)',
                                  
                                  titlefont = {"size": 15, "color":"White"},
                                  geo_scope='world',
                                  margin={"r":0,"t":40,"l":0,"b":0},
                                  paper_bgcolor='#000000',
                                 
                                  )
            )

col_map2.show()


# In[40]:


#Plotting on the WorldMap using plotly
x3 = df_country.groupby(["Country"])["Overweight"].mean()
col_map3 = go.Figure(
                data=go.Choropleth(
                locations = x3.index,
                locationmode = 'country names',
                colorscale= 'portland',
                text= x3.index,
                z=x3,
                colorbar = {'title':'Overweight %', 'len':200,'lenmode':'pixels' },
                
            ), layout = go.Layout(geo=dict(bgcolor= 'rgba(255,255,255,0)',showframe=False, projection = {'type': 'natural earth'}),
                                  title = 'Overweight % (Ukraine)',
                                  
                                  titlefont = {"size": 15, "color":"White"},
                                  geo_scope='world',
                                  margin={"r":0,"t":40,"l":0,"b":0},
                                  paper_bgcolor='#000000',
                                 
                                  )
            )

col_map3.show()


# In[41]:


#Plotting on the WorldMap using plotly
x4 = df_country.groupby(["Country"])["Stunting"].mean()
col_map4= go.Figure(
                data=go.Choropleth(
                locations = x4.index,
                locationmode = 'country names',
                colorscale= 'portland',
                text= x4.index,
                z=x4,
                colorbar = {'title':'Stunting%', 'len':200,'lenmode':'pixels' },
                
            ), layout = go.Layout(geo=dict(bgcolor= 'rgba(255,255,255,0)',showframe=False, projection = {'type': 'natural earth'}),
                                  title = 'Stunting %(Bangladesh)',
                                  
                                  titlefont = {"size": 15, "color":"White"},
                                  geo_scope='world',
                                  margin={"r":0,"t":40,"l":0,"b":0},
                                  paper_bgcolor='#000000',
                                 
                                  )
            )

col_map4.show()


# In[42]:


import plotly.graph_objects as go
x5 = df_country.groupby(["Country"])["Underweight"].mean()
col_map5  = go.Figure(
                data=go.Choropleth(
                locations = x5.index,
                locationmode = 'country names',
                colorscale= 'portland',
                text= x5.index,
                z=x5,
                colorbar = {'title':' Underweight %', 'len':200,'lenmode':'pixels' },
                
            ), layout = go.Layout(geo=dict(bgcolor= 'rgba(255,255,255,0)',showframe=False, projection = {'type': 'natural earth'}),
                                  title = 'Underweight % (Bangladesh)',
                                  
                                  titlefont = {"size": 15, "color":"White"},
                                  geo_scope='world',
                                  margin={"r":0,"t":40,"l":0,"b":0},
                                  paper_bgcolor='#000000',
                                 
                                  )
            )

col_map5.show()


# # Most malnourished countries

# In[43]:


df_country.sort_values(by=['Severe Wasting','Stunting','Wasting','Underweight','Overweight'], ascending =False).head(10)


# **Observation**:
#     
# South Sudan , Djibouti, India are top three Malnutrioned Affected Countries. They are more Low-Income,Lower-Middle-Income across Malnutrioned Countries

# # Countries which are very less affected from malnutrition

# In[44]:


df_country.sort_values(by=['Severe Wasting','Stunting','Wasting','Underweight','Overweight']).head(10)


# **Observation**:
#  Australia , USA, Germany Are less malnutrioned affected and as well High-Income Countries who can afford Three Meals per Day.  

# In[45]:


app = dash.Dash()
df = px.data.tips()# Build App


body = {
    'background-color':'#000000',
    'border': '5px solid black',
    'margin': '30px',
    'box-shadow': '5px 10px',
    'color': 'white',
    'font-family': "Lucida Sans"   
    
}



fig1=df_country.groupby(['Income Classification'])[['Severe Wasting','Wasting','Overweight','Stunting','Underweight']].mean().iplot(kind='bar',asFigure=True,title='Parameter mean across Incomes')


x1 = df_country.groupby(["Country"])["Severe Wasting"].mean()
x2 = df_country.groupby(["Country"])["Wasting"].mean()
x3 = df_country.groupby(["Country"])["Overweight"].mean()
x4 = df_country.groupby(["Country"])["Stunting"].mean()
x5 = df_country.groupby(["Country"])["Underweight"].mean()



app.layout = html.Div(id='Main-div',style=body, children=[
    html.Div(
    html.H1("MALNUTRITION-COUNTRY-ESTIMATES ",
            style={'text-align': 'center','font-family':'Helvetica' })),
    
     
     dcc.Graph(figure=col_map1),
     dcc.Graph(figure=col_map2),
     dcc.Graph(figure=col_map3),
     dcc.Graph(figure=col_map4),
     dcc.Graph(figure=col_map5),
     dcc.Graph(figure=fig1),
     dcc.Graph(figure=fig2),
     dcc.Graph(figure=fig3),
     dcc.Graph(figure=fig4),
     dcc.Graph(figure=fig5),
    
    
    

])







    


# In[46]:


if __name__ == '__main__':
    app.run_server(debug=True,port=8051,use_reloader=False)#debug=True,use_reloader=False


# In[ ]:




