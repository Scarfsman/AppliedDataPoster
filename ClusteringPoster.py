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
import shapefile as shp
import seaborn as sns
import numpy as np
import geopandas as gpd

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

def createClusters(df, label1, label2, n = 2):
    cluster_data = ct.scaler(df[[label1, label2]])
    cluster_df = cluster_data[0]

    #training the cluster model on the data we have
    ncluster = n
    kmeans = cluster.KMeans(n_clusters = ncluster)
    kmeans.fit(cluster_df)
    labels = kmeans.labels_
    df['Groups'] = labels

    fix, ax = plt.subplots()

    ax.scatter(df[label1], 
               df[label2],
               c = labels)

    plt.xlabel(label1)
    plt.ylabel(label2)
  
def read_shapefile(sf):
  #fetching the headings from the shape file
  fields = [x[0] for x in sf.fields][1:]#fetching the records from the shape file
  records = [list(i) for i in sf.records()]
  shps = [s.points for s in sf.shapes()]#converting shapefile data into pandas dataframe
  df = pd.DataFrame(columns=fields, data=records)#assigning the coordinates
  df = df.assign(coords=shps)
  return df  
  
#Reading in the data
CO2_df, CO2_df_tr = getDfs('World_Bank_Data/CO2.csv')
GDP_df, GDP_df_tr = getDfs('World_Bank_Data/GDP.csv')
MIG_df, MIG_df_tr = getDfs('World_Bank_Data/Migration.csv')
RET_df, RET_df_tr = getDfs('World_Bank_Data/Retired.csv')
HIG_df, HIG_df_tr = getDfs('World_Bank_Data/Higher.csv')

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

createClusters(EU_df, 'Migration', 'GDP')

#%%
#map test

sns.set(style='whitegrid', 
        palette='pastel', 
        color_codes=True) 

sns.mpl.rc('figure', figsize=(10,6))

shp_path = r"C:\Users\Georg\OneDrive\Desktop\NUTS_RG_20M_2021_3035.shp"
sf = shp.Reader(shp_path)

# =============================================================================
# map_df = read_shapefile(sf)
# map_df = map_df[map_df['LEVL_CODE'] == 0]
# map_df['coords'].plot()
# =============================================================================

fig, axel = plt.subplots()

map_df = gpd.read_file(shp_path)
map_df = map_df[map_df['LEVL_CODE'] == 0]
map_df_blorp = map_df[10:]
map_df_blorp.plot(ax = axel, color = 'white', edgecolor='black')
map_df_glrop = map_df[:10]
map_df_glrop.plot(ax = axel, color = 'red', edgecolor='black')
map_df_glrop = map_df[14:18]
map_df_glrop.plot(ax = axel, color = 'blue', edgecolor='black')
plt.xlim(left = 2500000, right = 7500000)
plt.ylim(bottom = 1000000, top = 5500000)
