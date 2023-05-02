# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:54:31 2023

@author: Georg
"""

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct

def getDfs(filename):
    """
    Parameters
    ----------
    dataframes : str()
        The wordbank dataset in CSV format to be read into the script
    Returns
    -------
    temp_df : pd.DataFrame
        The world bank data frame read in the script with countries as rows and
        years as columns
    temp_df_tr : pd.DataFrame
        The world bank data frame read in the script with years as rows and
        countries as columns
    """
    #skip the rows with no useful information
    temp_df = pd.read_csv(filename, skiprows = 4)
    
    temp_df_tr = pd.DataFrame(temp_df)
    temp_df_tr = temp_df_tr.transpose()
    temp_df_tr.columns = list(temp_df_tr.loc['Country Name',:])
    #clean up the transposed dataframe by dropping the wors that aren't of 
    #interest
    temp_df_tr = temp_df_tr.drop(['Country Name', 
                                  'Country Code',
                                  'Indicator Name',
                                  'Indicator Code'])
    
    #clean up the untrasnposed dataframe. this is so that both dataframes can
    #be passed to the functions defined below
    temp_df = temp_df.set_index('Country Name', drop = True)
    temp_df = temp_df.drop(['Country Code',
                            'Indicator Name',
                            'Indicator Code'],
                             axis = 1)
    
    return temp_df, temp_df_tr

#Reading in the data
CO2_df, CO2_df_tr = getDfs('CO2.csv')
GDP_df, GDP_df_tr = getDfs('GDP.csv')
MIG_df, MIG_df_tr = getDfs('Migration.csv')
RET_df, RET_df_tr = getDfs('Retired.csv')
HIG_df, HIG_df_tr = getDfs('Higher.csv')

#creating list to filter data by
EU_members = ['Austria', 
              'Belgium', 
              'Bulgaria', 
              'Croatia', 
              'Cyprus', 
              'Czechia', 
              'Denmark', 
              'Estonia', 
              'Finland', 
              'France', 
              'Germany',
              'Greece', 
              'Hungary', 
              'Ireland', 
              'Italy', 
              'Latvia', 
              'Lithuania',
              'Luxembourg',
              'Malta', 
              'Netherlands', 
              'Poland', 
              'Portugal', 
              'Romania', 
              'Slovak Republic', 
              'Slovenia', 
              'Spain',
              'Sweden']

#Creating dataframe to look at information related to EU memeber states 
#for the metrics of interest
GDP_EU_df = GDP_df.loc[EU_members, '2016']
CO2_EU_df = CO2_df.loc[EU_members, '2016']
MIG_EU_df = MIG_df.loc[EU_members, '2016']
RET_EU_df = RET_df.loc[EU_members, '2016']
HIG_EU_df = HIG_df.loc[EU_members, '2016']


EU_df = pd.DataFrame(GDP_EU_df)
EU_df['GDP'] = EU_df['2016']
EU_df = EU_df.drop('2016', axis = 1)
EU_df['CO2'] = CO2_EU_df
EU_df['Migration'] = MIG_EU_df 
EU_df['Retired'] = RET_EU_df 
EU_df['Higher'] = HIG_EU_df 

cluster_data = ct.scaler(EU_df[['GDP', 'Higher']])
cluster_df = cluster_data[0]

#calulating the optimal number of clusters
x = [x for x in range(2, 20)]
y = []

for i in x:
    ncluster = i
    kmeans = cluster.KMeans(n_clusters = ncluster)
    kmeans.fit(cluster_df)
    labels = kmeans.labels_
    y.append(skmet.silhouette_score(cluster_df, labels))

#creating the plot
fig, ax = plt.subplots()
plt.plot(x, y)

#training the cluster model on the data we have
ncluster = 2
kmeans = cluster.KMeans(n_clusters = ncluster)
kmeans.fit(cluster_df)
labels = kmeans.labels_

fix, ax = plt.subplots()

ax.scatter(EU_df['GDP'], 
           EU_df['Higher'],
           c = labels)

plt.xlabel('Net Migration')
plt.ylabel('GDP')

