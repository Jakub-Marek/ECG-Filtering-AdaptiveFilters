from scipy.misc import electrocardiogram
import numpy as np
import matplotlib.pyplot as plt
import padasip as pa 
from math import log10, sqrt



def ecg_LMS(target,signal):
  

    #LMS Filter:
    filt = pa.filters.AdaptiveFilter(model='LMS',n=3,mu=0.01,w='random')
    LMSy,e,w = filt.run(target,signal)
    
    mse_LMS = np.mean(e**2)
    print('MSE LMS = ', mse_LMS)
    psnr_LMS = 20*log10(2/sqrt(mse_LMS))
    print('PSNR for LMS = ',psnr_LMS)


    return LMSy



def ecg_NLMS(target,signal):


    #NLMS Filter:
    filt = pa.filters.AdaptiveFilter(model='NLMS',n=3,mu=0.01,w='random')
    NLMSy,e,w = filt.run(target,signal)

    mse_NLMS = np.mean(e**2)
    print('MSE NLMS = ', mse_NLMS)
    psnr_NLMS = 20*log10(2/sqrt(mse_NLMS))
    print('PSNR for NLMS = ',psnr_NLMS)


    return NLMSy



def ecg_RLSy(target,signal):

    #RLS Filter:
  
    filt = pa.filters.AdaptiveFilter(model='RLS',n=3,mu=0.01,w='random')
    RLSy,e,w = filt.run(target,signal)

    mse_RLSy = np.mean(e**2)
    print('MSE RLS = ', mse_RLSy)
    psnr_RLSy = 20*log10(2/sqrt(mse_RLSy))
    print('PSNR for RLS = ',psnr_RLSy)

    return RLSy


ecg = electrocardiogram()
short_ecg = (ecg[0:600])
w = np.random.uniform(0,1,[3,1])


short_ecg = (ecg[0:600])
signal = np.reshape(short_ecg,(200,3))
#Adding noise to ecg signal:
noise = np.random.normal(0,0.2,(600))
ecg_noise = short_ecg + noise
print(w)
target = w[0]*signal[:,0] + w[1]*signal[:,1] + w[2]*signal[:,2] + noise[0:200]


LMSy = ecg_LMS(target,signal)
NLMSy = ecg_NLMS(target,signal)
RLSy = ecg_RLSy(target,signal)


plt.plot(short_ecg)
plt.title('ECG Signal')
plt.show()

plt.plot(ecg_noise)
plt.title('ECG Noise Signal')
plt.show()

plt.plot(signal)
plt.title('ECG 2-D Array Signal')
plt.show()

plt.plot(target)
plt.title('ECG Target Signal')
plt.show()

plt.plot(np.abs(LMSy))
plt.title('ECG LMS Filtered Signal')
plt.show()

plt.plot(np.abs(NLMSy))
plt.title('ECG NLMS Filtered Signal')
plt.show()

plt.plot(np.abs(RLSy))
plt.title('ECG RLS Filtered Signal')
plt.show()