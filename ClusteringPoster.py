# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:54:31 2023

@author: Georg
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import cluster_tools as ct
import errors as err
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

def createClusters(label1, label2, year):
    """
    Parameters
    ----------
    label1 : The first parameter for generating clusters
    
    label2 : the second label for generating clusters
    
    year : the year to generate clusters for
        
    Returns
    -------
    Generates a scatter graph showing a clusters for EU members for the passed
    parameters in the given year
    """
    
    #list of EU members with their world bank and EU shpae file labels
    #Used to filter the world bank data and the shape file in order to plot
    #the countries
    WB2EU = {'Austria' : 'AT', 
             'Belgium' : 'BE', 
             'Bulgaria' : 'BG', 
             'Croatia' : 'HR', 
             'Cyprus' : 'CY', 
             'Czechia' : 'CZ', 
             'Denmark' : 'DK', 
             'Estonia': 'EE', 
             'Finland': 'FI', 
             'France' : 'FR', 
             'Germany' : 'DE',
             'Greece' : 'EL', 
             'Hungary' : 'HU', 
             'Ireland' : 'IE', 
             'Italy' : 'IT', 
             'Latvia' : 'LV', 
             'Lithuania' : 'LT',
             'Luxembourg' : 'LU',
             'Malta' : 'MT', 
             'Netherlands' : 'NL', 
             'Poland' : 'PL', 
             'Portugal' : 'PT', 
             'Romania' : 'RO', 
             'Slovak Republic' : 'SK', 
             'Slovenia' : 'SI', 
             'Spain' : 'ES',
             'Sweden' : 'SE',
             'United Kingdom' : 'UK'}
    
    #Generate the dataframe for our indicators for a given year
    GDP_EU_df = GDP_df.loc[list(WB2EU.keys()), year]
    CO2_EU_df = CO2_df.loc[list(WB2EU.keys()), year]
    MIG_EU_df = MIG_df.loc[list(WB2EU.keys()), year]
    RET_EU_df = RET_df.loc[list(WB2EU.keys()), year]
    HIG_EU_df = HIG_df.loc[list(WB2EU.keys()), year]
    URB_EU_df = URB_df.loc[list(WB2EU.keys()), year]

    df = pd.DataFrame(GDP_EU_df)
    df['GDP'] = df[year]
    df = df.drop(year, axis = 1)
    df['CO2'] = CO2_EU_df
    df['Migration'] = MIG_EU_df 
    df['Retired'] = RET_EU_df 
    df['Higher'] = HIG_EU_df 
    df['Urban'] = URB_EU_df 

    #noramlise the data
    cluster_data = ct.scaler(df[[label1, label2]])
    cluster_df = cluster_data[0]

    #training the cluster model on the data we have
    ncluster = 2
    kmeans = cluster.KMeans(n_clusters = ncluster)
    kmeans.fit(cluster_df)
    labels = kmeans.labels_
    df['Groups'] = labels

    #Pot the two groups we have
    fix, ax = plt.subplots()
    
    temp = df[df['Groups'] == 0]
    ax.scatter(temp[label1], 
               temp[label2],
               c = 'purple',
               edgecolors = 'Black')
    
    temp = df[df['Groups'] == 1]
    ax.scatter(temp[label1], 
               temp[label2],
               c = 'yellow',
               edgecolors = 'black')
    
    back = ct.backscale(kmeans.cluster_centers_, 
                        cluster_data[1], 
                        cluster_data[2])
    
    #plot the cluster centres
    ax.scatter(back[:,0], 
               back[:,1], 
               c='white', 
               marker = 'd',
               edgecolor = 'black',
               s = 150)
    
    plt.ylim(top = 25, bottom = 5)
    plt.xlim(left = 10, right = 150)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title(year)
    fname = 'scatter_' + year + '.png'
    plt.savefig(fname)
    
    #Generates the Map
    
    sns.set(style='white', 
            palette='pastel', 
            color_codes=True) 
    sns.mpl.rc('figure', figsize=(10,6))
    shp_path = "Shape_File/NUTS_RG_20M_2021_3035.shp"
    fig, axel = plt.subplots()

    map_df = gpd.read_file(shp_path)
    map_df = map_df[map_df['LEVL_CODE'] == 0]
    map_df = map_df.set_index('NUTS_ID')
    map_df.plot(ax = axel, color = 'White', edgecolor='black')

    Group1_Codes = [WB2EU[i] for i in df[df['Groups'] == 0].index]
    temp = map_df.loc[Group1_Codes]
    temp.plot(ax = axel, 
              color = 'Purple', 
              edgecolor='black', )
    
    print([i for i in df[df['Groups'] == 0].index])
    print([i for i in df[df['Groups'] == 1].index])

    Group2_Codes = [WB2EU[i] for i in df[df['Groups'] == 1].index]
    temp = map_df.loc[Group2_Codes]
    temp.plot(ax = axel, 
              color = 'Yellow', 
              edgecolor='black')

    #creating artists for the legend
    pp = mpatches.Patch(color ='purple', 
                               label ='Group 1')

    yp = mpatches.Patch(color ='yellow', 
                                label ='Group 2',
                                edgecolor = 'black')

    plt.legend(handles=[pp, yp])
    plt.xlim(left = 2500000, right = 7500000)
    plt.ylim(bottom = 1000000, top = 5500000)
    plt.xticks(color='w')
    plt.yticks(color='w')
    plt.title(year)
    fname = 'map_' + year + '.png'
    plt.savefig(fname)
    
#Reading in the data
CO2_df, CO2_df_tr = getDfs('World_Bank_Data/CO2.csv')
GDP_df, GDP_df_tr = getDfs('World_Bank_Data/GDP.csv')
MIG_df, MIG_df_tr = getDfs('World_Bank_Data/Migration.csv')
RET_df, RET_df_tr = getDfs('World_Bank_Data/Retired.csv')
HIG_df, HIG_df_tr = getDfs('World_Bank_Data/Higher.csv')
URB_df, URB_df_tr = getDfs('World_Bank_Data/Urban_pop.csv')

#Creating dataframe to look at information related to EU memeber states 
#for the metrics of interest

createClusters('Higher', 'Retired', '1995')
createClusters('Higher', 'Retired', '2018')

#%%

fig, ax = plt.subplots()

target_df = URB_df_tr.drop('Unnamed: 66')

#RET_df_tr = RET_df_tr.drop('Unnamed: 66')

ax.scatter(target_df.index, target_df['European Union'])

plt.xlabel('Year')
plt.xticks(rotation = 90)

def linear(t, s, k, t0):
    t = t-t0
    f = s + k*t
    return f

def poly(t, c0, c1, c2, c3, t0):
    t = t - t0
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = [func(i, *param) for i in x]
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
    
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = [func(i, *p) for i in x]
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper, pmix  

years = [int(i) for i in list(target_df.index)]
initparam = [60, 1, 1, 1, 1960]
param, covar = opt.curve_fit(poly, 
                             years, 
                             target_df['European Union'],
                             p0 = initparam)

pop_est = [poly(i, *param) for i in years]
ax.plot(target_df.index, pop_est, color = 'black')

sigma = np.sqrt(np.diag(covar))

#errorLow, errorUp, uplow = err_ranges(years, poly, param, sigma)
#plt.fill_between(target_df.index, errorLow, errorUp, alpha = 0.7)

# =============================================================================
# years = [int(i) for i in list(group2_df.index)]
# param, covar = opt.curve_fit(poly, years, group2_df['mean'])
# pop_est = [poly(i, *param) for i in years]
# ax.plot(group2_df.index, pop_est, color = 'black')
# =============================================================================

