# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:54:31 2023

@author: Georg
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.cluster as cluster
import scipy.optimize as opt
import cluster_tools as ct
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

def createClusters(label1, label2, year, xlabel, ylabel):
    """
    Parameters
    ----------
    label1 : The first parameter for generating clusters
    
    label2 : the second label for generating clusters
    
    year : the year to generate clusters for
    
    xlabel : the label for the x axis
    
    ylabel : the label for the y axis
        
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
               edgecolors = 'Black',
               label = 'Group 1')
    
    temp = df[df['Groups'] == 1]
    ax.scatter(temp[label1], 
               temp[label2],
               c = 'yellow',
               edgecolors = 'black',
               label = 'Group 2')
    
    back = ct.backscale(kmeans.cluster_centers_, 
                        cluster_data[1], 
                        cluster_data[2])
    
    #plot the cluster centres
    ax.scatter(back[:,0], 
               back[:,1], 
               c='white', 
               marker = 'd',
               edgecolor = 'black',
               s = 150,
               label = 'Cluster Centres')
    
    plt.ylim(top = 25, bottom = 5)
    plt.xlim(left = 10, right = 150)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Demographic Clusters in ' + year, fontsize = 17)
    plt.legend(loc = "upper left")
    fname = 'scatter_' + year + '.png'
    plt.savefig(fname, bbox_inches='tight')
    
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
    axel.ticklabel_format(style='plain')
    fname = 'map_' + year + '.png'
    plt.savefig(fname, bbox_inches='tight')
        
def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    
    Had issues with passing the a numpy array, I've altered it slightly to
    accept lists'
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

def fitEUData(func, initparams, df, title = '', ylabel = ''):
    '''Generates a graph for the EU data fitted to the function func for the
    dataframe df from 1990 to 2021. df must be tr. title and ylabel refer to 
    plt attrributes of the same name. Projects data to 2030'''
    
    #Create the dataframe for the fitting portion of the poster
    target_df = df.loc['1990':].drop('Unnamed: 66')

    #fit the function
    #NOTE: I've fitted the function on a list starting from 0 rather then passing
    # it the year and and doing t = t- t0 in the function, as doing it the later 
    # way resulted in 12 figure covariances, which did not produce meaningful
    # confidence intervals.
    
    years = [int(i) for i in range(0, len(target_df.index))]
    param, covar = opt.curve_fit(func, 
                                 years, 
                                 target_df['European Union'],
                                 p0 = initparams)
    
    #get uncertainty for 1990 to 2030
    projection = []
    
    for i in range(1990, 2031):
        projection.append(str(i))
        
    years = [int(i) for i in range(0, len(projection))] 
    
    pop_est = [func(i, *param) for i in years]
    sigma = np.sqrt(np.diag(covar))
    errorLow, errorUp, uplow = err_ranges(years, func, param, sigma)

    #Plotting the data
    fig, ax = plt.subplots()

    #plotting the fitted function
    ax.plot(projection, 
            pop_est, 
            color = 'black',
            label = 'Fitted Function')

    #plotting the lower bound of the confidence interval
    ax.plot(projection, 
            errorLow, 
            color = 'black',
            alpha = 0.3)

    #plotting the upper bound of the confidence interval
    ax.plot(projection, 
            errorUp, 
            color = 'black',
            alpha = 0.3)

    #filling the sapce between the points
    ax.fill_between(projection, 
                    errorLow, 
                    errorUp, 
                    alpha = 0.2, 
                    color = 'yellow',
                    label = 'Confidence Interval')
    
    #plotting the data points
    ax.scatter(target_df.index, 
               target_df['European Union'],
               edgecolor = 'black',
               color = 'purple',
               label = 'Observed Data Points')
    
    ax.axvline('2021',
               color = 'black',
               linestyle = '--',
               label = 'Projection',
               alpha = 0.75)

    #labels
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.title(title, fontsize = 17)
    plt.xticks(rotation = 90)
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 5 != 0:
            label.set_visible(False)
    plt.legend()
    
    #save the figure
    plt.savefig('FittedData.png')

#define our function to fit
def poly(t, c0, c1, c2, c3):
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f  

#Reading in the data
CO2_df, CO2_df_tr = getDfs('World_Bank_Data/CO2.csv')
GDP_df, GDP_df_tr = getDfs('World_Bank_Data/GDP.csv')
MIG_df, MIG_df_tr = getDfs('World_Bank_Data/Migration.csv')
RET_df, RET_df_tr = getDfs('World_Bank_Data/Retired.csv')
HIG_df, HIG_df_tr = getDfs('World_Bank_Data/Higher.csv')
URB_df, URB_df_tr = getDfs('World_Bank_Data/Urban_pop.csv')

#Create the clusters for the years of interest
xlab = 'School enrollment, tertiary (% gross)'
ylab = 'Population ages 65 and above (% of total population)'
createClusters('Higher', 'Retired', '1995', xlab, ylab)
createClusters('Higher', 'Retired', '2018', xlab, ylab)

#fitting the data to our function
initparams = [14, 0.25, 0.1, 0.1]
fitEUData(poly,
          initparams,
          RET_df_tr,
          title = 'Percent of People in the EU above 64 Years of Age',
          ylabel = ylab)
            