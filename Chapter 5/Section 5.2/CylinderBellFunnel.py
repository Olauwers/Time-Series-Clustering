# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:44:28 2019

@author: olauwers
"""

import numpy as np
import pandas as pd
import scipy.signal as sps
from scipy.stats import zscore

import matplotlib.pyplot as plt

from pyts.datasets import make_cylinder_bell_funnel
from sklearn.cluster import KMeans
import spectrum as spc

# %%

def power_cepstrum(y,windowsize,fs):
    # power_cepstrum expects the time domain information of input (u) and output (y), as it will calculate the
    # power spectral density (psd) using Welch's method, which provides a better estimate of the psd
    # than the naive fft-implementation. power_cepstrum then returns the power cepstrum of the underlying model.
    # The subtracton of output and input power cepstrum is again changed into a division in the frequency 
    # domain (see complex_cepstrum).
    
#    Pyy = np.abs(np.fft.fft(y,windowsize)) ** 2
 
#    f, Pyy = sps.welch(y,fs,nperseg=windowsize,nfft = 2**20,return_onesided=True)  # Estimate the psd's of u and y.
#    f, Pyy = sps.welch(y,fs,nperseg=windowsize,return_onesided=True)  # Estimate the psd's of u and y.
#    powerceps = np.real(np.fft.irfft(np.log(Pyy,out=np.zeros_like(Pyy), where=(Pyy!=0))))  
#    
    Sk_complex, weights, eigenvalues = spc.mtm.pmtm(y,NW=12,show = False)
    Sk = np.abs(Sk_complex)**2
    Pyy = np.mean(Sk * weights.T, axis=0)    
#    powerceps = np.real(np.fft.ifft(np.log(Pyy,out=np.zeros_like(Pyy), where=(Pyy!=0))))                      # Estimate the power cepstrum.
    powerceps = np.real(np.fft.ifft(np.log(Pyy)))                      # Estimate the power cepstrum.
#    
    return powerceps

def rolling_window(a, window, step_size):
    shape = (np.int((a.shape[-1]-window)/step_size)+1,window)
    rolled = np.empty(shape)
    
    for i in np.arange(shape[0]):
        rolled[i,:] = a[i*step_size: i*step_size + window]
    
    return rolled

# %% Cylinder_bell_funnel data --> CHOOSE WHETHER YOU WANT TWO OR THREE CLUSTERS
    
number_of_samples = 12

proportion = (1/2,1/2)     #For only cylinder and bell shapes
amount_of_clusters = 2

#proportion = (1/3,1/3)      #For all three shapes
#amount_of_clusters = 3

data, labels = make_cylinder_bell_funnel(n_samples = number_of_samples, shuffle = True, weights = proportion)

fs = 1
length_of_window = 128

timeseries = np.hstack(data)

np.savetxt('cylinderbellfunneltimeseries.csv',np.concatenate((np.reshape(np.arange(timeseries.size),(timeseries.size,1)),np.reshape(timeseries,(timeseries.size,1))),axis=1),delimiter = ',')

windowed_data = rolling_window(timeseries,length_of_window,1)



# %% Euclidean clustering

kmeanseuclid = KMeans(n_clusters = amount_of_clusters,n_init=1000).fit(np.nan_to_num(zscore(windowed_data,axis=1)))


# %% Plot cluster labels vs. ground truth

labels_per_point = np.vstack(length_of_window * [labels]).T.reshape((length_of_window*labels.size,))[length_of_window//2:-length_of_window//2]
plt.figure()
plt.plot(kmeanseuclid.labels_,'b',label='Clustering results')
plt.plot(labels_per_point,'r:',label='Ground truth')
plt.xlabel('Window number')
plt.ylabel('Cluster label')
plt.title('Euclidean k-means')
plt.legend()
plt.show()
np.unique(kmeanseuclid.labels_,return_counts = True)

#%% Plot cluster averages

plt.figure()
plt.plot(kmeanseuclid.cluster_centers_[0],label='Cluster 0')
plt.plot(kmeanseuclid.cluster_centers_[1],label='Cluster 1')

if amount_of_clusters >= 3:
    plt.plot(kmeanseuclid.cluster_centers_[2],label='Cluster 2')
    
plt.xlabel('k')
plt.ylabel('Average of cluster signals')
plt.title('Cluster centers of Euclidean k-means')
plt.legend()
plt.show()



#%% Save

#np.savetxt('groundtruth.csv',np.concatenate((np.reshape(np.arange(kmeans.labels_.size),(kmeans.labels_.size,1)),np.reshape(labels_per_point,(kmeans.labels_.size,1))),axis=1),delimiter = ',')

#np.savetxt('euclideanlabels.csv',np.concatenate((np.reshape(np.arange(kmeanseuclid.labels_.size),(kmeanseuclid.labels_.size,1)),np.reshape(kmeanseuclid.labels_,(kmeanseuclid.labels_.size,1))),axis=1),delimiter = ',')

#np.savetxt('euclideanclustercenter0.csv',np.concatenate((np.reshape(np.arange(length_of_window),(length_of_window,1)),np.reshape(kmeanseuclid.cluster_centers_[0],(length_of_window,1))),axis=1),delimiter = ',')
#np.savetxt('euclideanclustercenter1.csv',np.concatenate((np.reshape(np.arange(length_of_window),(length_of_window,1)),np.reshape(kmeanseuclid.cluster_centers_[1],(length_of_window,1))),axis=1),delimiter = ',')
#np.savetxt('euclideanclustercenter2.csv',np.concatenate((np.reshape(np.arange(length_of_window),(length_of_window,1)),np.reshape(kmeanseuclid.cluster_centers_[2],(length_of_window,1))),axis=1),delimiter = ',')


# %% Cepstral clustering

length_of_cepstra = 2**7
number_of_differences = 0
cutoff = 32


windowed_cepstra = np.asarray([power_cepstrum(np.diff(windowed_data,n=number_of_differences,axis=1)[i],length_of_cepstra,fs) for i in np.arange(windowed_data.shape[0])])
weights = np.sqrt(np.arange(cutoff))
#weights[0] = 1
windowed_weighted_cepstra = np.asarray([weights * windowed_cepstra[i][:cutoff] for i in np.arange(windowed_data.shape[0])])

kmeansceps = KMeans(n_clusters = amount_of_clusters,n_init=1000).fit(np.nan_to_num(windowed_weighted_cepstra))

# %% Plot cepstral cluster labels vs. ground truth

labels_per_point = np.vstack(length_of_window * [(labels)]).T.reshape((length_of_window*labels.size,))[length_of_window//2:-length_of_window//2+1]
plt.figure()
plt.plot(kmeansceps.labels_,'b',label = 'Clustering results')
plt.plot(labels_per_point,'r:',label='Ground truth')
plt.xlabel('Window number')
plt.ylabel('Cluster label')
plt.title('Cepstral k-means')
plt.legend()
plt.show()
np.unique(kmeansceps.labels_,return_counts = True)

plt.figure()
plt.plot(np.mean(windowed_data[kmeansceps.labels_==0],axis=0),label='Cluster 0')
plt.plot(np.mean(windowed_data[kmeansceps.labels_==1],axis=0),label='Cluster 1')

if amount_of_clusters >= 3:
    plt.plot(np.mean(windowed_data[kmeansceps.labels_==2],axis=0),label='Cluster 2')
    
plt.xlabel('k')
plt.ylabel('Average of cluster signals')
plt.title('Average of signals in a cluster for cepstral k-means')
plt.legend()
plt.show


#%% Plot average cepstra

plt.figure()
plt.plot(kmeansceps.cluster_centers_[0],label='Cluster 0')
plt.plot(kmeansceps.cluster_centers_[1],label='Cluster 1')

if amount_of_clusters >= 3:
    plt.plot(kmeansceps.cluster_centers_[2],label='Cluster 2')
    
plt.xlabel('k')
plt.ylabel('Cluster center value')
plt.title('Cluster centers of cepstral k-means')
plt.legend()
plt.show()

#%% Save

#np.savetxt('groundtruth2clusters.csv',np.concatenate((np.reshape(np.arange(kmeansceps.labels_.size),(kmeansceps.labels_.size,1)),np.reshape(labels_per_point,(kmeansceps.labels_.size,1))),axis=1),delimiter = ',')
#np.savetxt('cepstrallabels2clusters.csv',np.concatenate((np.reshape(np.arange(kmeansceps.labels_.size),(kmeansceps.labels_.size,1)),np.reshape(kmeansceps.labels_,(kmeansceps.labels_.size,1))),axis=1),delimiter = ',')


#np.savetxt('cepstrallabels.csv',np.concatenate((np.reshape(np.arange(kmeansceps.labels_.size),(kmeansceps.labels_.size,1)),np.reshape(kmeansceps.labels_,(kmeansceps.labels_.size,1))),axis=1),delimiter = ',')

#np.savetxt('weightedcepstralclustercenter0.csv',np.concatenate((np.reshape(np.arange(kmeansceps.cluster_centers_[0].size),(kmeansceps.cluster_centers_[0].size,1)),np.reshape(kmeansceps.cluster_centers_[0],(kmeansceps.cluster_centers_[0].size,1))),axis=1),delimiter = ',')
#np.savetxt('weightedcepstralclustercenter1.csv',np.concatenate((np.reshape(np.arange(kmeansceps.cluster_centers_[0].size),(kmeansceps.cluster_centers_[0].size,1)),np.reshape(kmeansceps.cluster_centers_[1],(kmeansceps.cluster_centers_[0].size,1))),axis=1),delimiter = ',')
#np.savetxt('weightedcepstraclustercenter2.csv',np.concatenate((np.reshape(np.arange(length_of_window),(length_of_window,1)),np.reshape(kmeansceps.cluster_centers_[2],(length_of_window,1))),axis=1),delimiter = ',')
#np.savetxt('averagewindowcepstralclustercenter0.csv',np.concatenate((np.reshape(np.arange(length_of_window),(length_of_window,1)),np.reshape(np.mean(windowed_data[kmeansceps.labels_==0],axis=0),(length_of_window,1))),axis=1),delimiter = ',')
#np.savetxt('averagewindowcepstralclustercenter1.csv',np.concatenate((np.reshape(np.arange(length_of_window),(length_of_window,1)),np.reshape(np.mean(windowed_data[kmeansceps.labels_==1],axis=0),(length_of_window,1))),axis=1),delimiter = ',')
#np.savetxt('averagewindowcepstralclustercenter2.csv',np.concatenate((np.reshape(np.arange(length_of_window),(length_of_window,1)),np.reshape(np.mean(windowed_data[kmeans.labels_==2],axis=0),(length_of_window,1))),axis=1),delimiter = ',')






