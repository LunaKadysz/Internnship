# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 10:11:16 2021

@author: WOLFSOL
"""
import numpy as np
import seaborn as sns
from scipy import signal
from scipy.optimize import curve_fit
import cmath
from lmfit import Model
import hdf5storage
import json
import sys
#sys.path.insert(1, '../Luna/BIU_interniship')
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

# %%


data = hdf5storage.loadmat('data/9000_data.mat')
# %%
print(data.keys())


# %%
def MyLorentzian2(x,lz1_center,lz1_sigma,lz1_amplitude_real,lz1_amplitude_imag,lz2_center,lz2_sigma,lz2_amplitude_real,lz2_amplitude_imag,lz3_center,lz3_sigma,lz3_amplitude_real,lz3_amplitude_imag,lz4_center,lz4_sigma,lz4_amplitude_real,lz4_amplitude_imag,lz5_center,lz5_sigma,lz5_amplitude_real,lz5_amplitude_imag,lz6_center,lz6_sigma,lz6_amplitude_real,lz6_amplitude_imag,lz7_center,lz7_sigma,lz7_amplitude_real,lz7_amplitude_imag,lz8_center,lz8_sigma,lz8_amplitude_real,lz8_amplitude_imag,lz9_center,lz9_sigma,lz9_amplitude_real,lz9_amplitude_imag,lz10_center,lz10_sigma,lz10_amplitude_real,lz10_amplitude_imag,lz11_center,lz11_sigma,lz11_amplitude_real,lz11_amplitude_imag,lz12_center,lz12_sigma,lz12_amplitude_real,lz12_amplitude_imag,lz13_center,lz13_sigma,lz13_amplitude_real,lz13_amplitude_imag,lz14_center,lz14_sigma,lz14_amplitude_real,lz14_amplitude_imag,lz15_center,lz15_sigma,lz15_amplitude_real,lz15_amplitude_imag,lz16_center,lz16_sigma,lz16_amplitude_real,lz16_amplitude_imag,lz17_center,lz17_sigma,lz17_amplitude_real,lz17_amplitude_imag,lz18_center,lz18_sigma,lz18_amplitude_real,lz18_amplitude_imag,lz19_center,lz19_sigma,lz19_amplitude_real,lz19_amplitude_imag,lz20_center,lz20_sigma,lz20_amplitude_real,lz20_amplitude_imag):
    """returns the absolute values of the sum of 13 lorentzian function"""
    
    lz1_amplitude =  lz1_amplitude_real + 1j*lz1_amplitude_imag
    lz2_amplitude =  lz2_amplitude_real + 1j*lz2_amplitude_imag
    lz3_amplitude =  lz3_amplitude_real + 1j*lz3_amplitude_imag
    lz4_amplitude =  lz4_amplitude_real + 1j*lz4_amplitude_imag
    lz5_amplitude =  lz5_amplitude_real + 1j*lz5_amplitude_imag
    lz6_amplitude =  lz6_amplitude_real + 1j*lz6_amplitude_imag
    lz7_amplitude =  lz7_amplitude_real + 1j*lz7_amplitude_imag
    lz8_amplitude =  lz8_amplitude_real + 1j*lz8_amplitude_imag
    lz9_amplitude =  lz9_amplitude_real + 1j*lz9_amplitude_imag
    lz10_amplitude =  lz10_amplitude_real + 1j*lz10_amplitude_imag
    lz11_amplitude =  lz11_amplitude_real + 1j*lz11_amplitude_imag
    lz12_amplitude =  lz12_amplitude_real + 1j*lz12_amplitude_imag
    lz13_amplitude =  lz13_amplitude_real + 1j*lz13_amplitude_imag
    lz14_amplitude =  lz14_amplitude_real + 1j*lz14_amplitude_imag
    lz15_amplitude =  lz15_amplitude_real + 1j*lz15_amplitude_imag
    lz16_amplitude =  lz16_amplitude_real + 1j*lz16_amplitude_imag
    lz17_amplitude =  lz17_amplitude_real + 1j*lz17_amplitude_imag
    lz18_amplitude =  lz18_amplitude_real + 1j*lz18_amplitude_imag
    lz19_amplitude =  lz19_amplitude_real + 1j*lz19_amplitude_imag
    lz20_amplitude =  lz20_amplitude_real + 1j*lz20_amplitude_imag


    l1 = (lz1_amplitude * lz1_sigma**2) / ( lz1_sigma**2 + ( x - lz1_center )**2) 
    l2 = (lz2_amplitude * lz2_sigma**2) / ( lz2_sigma**2 + ( x - lz2_center )**2) 
    l3 = (lz3_amplitude * lz3_sigma**2) / ( lz3_sigma**2 + ( x - lz3_center )**2)
    l4 = (lz4_amplitude * lz4_sigma**2) / ( lz4_sigma**2 + ( x - lz4_center )**2) 
    l5 = (lz5_amplitude * lz5_sigma**2) / ( lz5_sigma**2 + ( x - lz5_center )**2)
    l6 = (lz6_amplitude * lz6_sigma**2) / ( lz6_sigma**2 + ( x - lz6_center )**2) 
    l7 = (lz7_amplitude * lz7_sigma**2) / ( lz7_sigma**2 + ( x - lz7_center )**2) 
    l8 = (lz8_amplitude * lz8_sigma**2) / ( lz8_sigma**2 + ( x - lz8_center )**2)
    l9 = (lz9_amplitude * lz9_sigma**2) / ( lz9_sigma**2 + ( x - lz9_center )**2) 
    l10 = (lz10_amplitude * lz10_sigma**2) / ( lz10_sigma**2 + ( x - lz10_center )**2) 
    l11 = (lz11_amplitude * lz11_sigma**2) / ( lz11_sigma**2 + ( x - lz11_center )**2) 
    l12 = (lz12_amplitude * lz12_sigma**2) / ( lz12_sigma**2 + ( x - lz12_center )**2) 
    l13 = (lz13_amplitude * lz13_sigma**2) / ( lz13_sigma**2 + ( x - lz13_center )**2)
    l14 = (lz14_amplitude * lz14_sigma**2) / ( lz14_sigma**2 + ( x - lz14_center )**2) 
    l15 = (lz15_amplitude * lz15_sigma**2) / ( lz15_sigma**2 + ( x - lz15_center )**2)
    l16 = (lz16_amplitude * lz16_sigma**2) / ( lz16_sigma**2 + ( x - lz16_center )**2) 
    l17 = (lz17_amplitude * lz17_sigma**2) / ( lz17_sigma**2 + ( x - lz17_center )**2) 
    l18 = (lz18_amplitude * lz18_sigma**2) / ( lz18_sigma**2 + ( x - lz18_center )**2)
    l19 = (lz19_amplitude * lz19_sigma**2) / ( lz19_sigma**2 + ( x - lz19_center )**2) 
    l20 = (lz20_amplitude * lz20_sigma**2) / ( lz20_sigma**2 + ( x - lz20_center )**2) 
    
    return np.abs(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12+l13+l14+l15+l16+l17+l18+l19+l20)

# %%

import datetime

p20 = [5039.9748,5509.97245,5879.9706,6469.9676500000005,6549.96725,6829.9658500000005,7139.9643,7499.9625,8059.9597,8119.9594,8829.95585,8949.95525,9819.9509,10319.9484,10489.94755,11909.94045,12769.93615,12869.93565,14099.9295,14819.9259]

Fc_low = 13300
Fc_high = 3000
v = data['ZZ'][:,0,0]
L = len(v)
Fs = 199999


results = {}

for i in range(40,80):
    if  i % 4 == 0 :
        cut_file = i
        results2 = results.copy()
        results = {}
    for j in range(118):
        print(i,j)

        if f'[{i},{j}]' not in results.keys():
            print(f'[{i},{j}]')
  
            v = data['ZZ'][:,i,j]

            sos_l = signal.bessel(5,Fc_low, 'low', fs=Fs,output='sos') # LPF filtering parameters
            sos_h = signal.bessel(5,Fc_high, 'high', fs=Fs,output='sos') # HPF filtering parameters

            x = signal.sosfilt(sos_l,v) # LPF
            x = signal.sosfilt(sos_h,x) # HPF

            freqs=np.fft.fftfreq(L,1/Fs)
            f = freqs[freqs>100] #frequency vector (x-axis in frec domain)
            fft= np.fft.fft(x)[freqs>100]

            if np.abs(max(fft)) >20000:
                results[f'[{i},{j}]'] = {'params': [] , 'red_chi': [],'chi_sqr': []}
                print('Nuevo:',i,j,np.abs(max(fft)))
                p2=p20

                mod = Model(MyLorentzian2)
                pars = mod.make_params()
                for n,p in enumerate(p2):
                    index_p = np.where(f>p)[0][0]
                    prefix = 'lz%d_' % (n+1)
                    pars.add(name = f'{prefix}center', value = f[index_p],vary=False)
                    mod.set_param_hint(f'{prefix}center', vary=False)
                    pars.add(name = f'{prefix}sigma',value =  5, min=0)
                    pars.add(name = f'{prefix}amplitude_real', value = fft[index_p].real)
                    pars.add(name = f'{prefix}amplitude_imag', value = fft[index_p].imag)

                t1 = datetime.datetime.now()
                result = mod.fit(np.abs(fft), pars, x=f,nan_policy='omit')
                t2 = datetime.datetime.now()
                params_ = list(result.values.values())
                results[f'[{i},{j}]']['params'] = list(params_)
                results[f'[{i},{j}]']['red_chi']= result.redchi
                results[f'[{i},{j}]']['chi_sqr'] = result.chisqr

                print(f'Time: {t2-t1}')
                with open(f'new_results/{cut_file}.txt', 'w') as file:
                    file.truncate(0)
                    file.write(json.dumps(results)) # use `json.loads` to do the reverse
                
                
    
# %%

# %%