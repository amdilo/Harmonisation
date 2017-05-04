#!/usr/bin/env python

"""FCDR harmonisation modules
    Project:        H2020 FIDUCEO
    Author:         Arta Dilo \NPL MM
    Date created:   07-03-2017
    Last update:    12-04-2017
    Version:        12.0
Plot results of Monte Carlo (MC) ODR harmonisation. """

from numpy import divide, loadtxt, sqrt 
from random import sample
import visFun as vis


""" Heat maps of coeffs correlation for ODR on simulated data and MC trials.
Arguments of the function:
    - fname: text file with fit coefficients in MC trials
    - slab: label of the sensor being fitted
    - beta: ODR fit coefficients on simulated data
    - cov: ODR evaluated covariance matrix of fit coefficients
    - mcm: MC method, full error structure or random errors 
"""
def mcCorr(fname, series, beta, cov, mcm):
    filelist = series.splist # list of nc files of matchups per pair
    
    # plot correlation of ODR coeffs on simulated data
    cor_ttl = 'ODR correlation of harmonisation coefficients for pairs\n'+', '.join(filelist)
    cor = vis.cov2cor(cov) # correlation matrix of ODR fit coeffs
    vis.plot_corr_heatmap(cor, title=cor_ttl, labels=['']) # correlations heat map  

    mcb = loadtxt(fname, delimiter=',') # get beta coeffs in MC trials
    q = mcb.shape[1] # number of columns: beta params + 1
    
    # compute mean, covariance and correlation of MC coeffs
    mcmean,mccov,mccor = vis.mcStats(series, mcb[:,0:q-1], beta, cov) 
    cor_ttl = 'MC correlation of harmonisation coefficients for pairs\n'+', '.join(filelist)
    vis.plot_corr_heatmap(mccor, title=cor_ttl, labels=['']) # correlations heat map  
    
    return mccov # return covariance of calib.coeffs in MC trials 


""" Graphs of radiance bias with sigma error bars for MC trials. 
TODO! modify for multiple pairs!
Arguments of the function:
    - avhrrS: instance of class AVHRR series 
    - noMU: number of matchups in the current pair
    - nobj: sample size of select records for plotting
    - inC: input calibration coefficients for the sensor
    - slab: label of the sensor being fitted
    - muTime: matchup times
    - Hrnd: matrix of random uncertainties of harmonisation variables
    - Hsys: matrix of systematic uncertainties of harmonisation vars
    - odrSD: ODR full output in fitting simulated data
    - MCcov: covariance matrix of MC calibration coefficients 
    - k: k-sigma range for uncertainties
    - mclab: lable for the error type of MC trials, random or error structure
"""
def plotMCres(avhrrS,noMU,nobj,inC,slab,Hrnd,Hsys,odrSD,MCcov,k,mclab):

    # compile data for graphs
    bodr = odrSD.beta # odr fit coefficients
    covodr = odrSD.cov_beta # odr evaluated covariance matrix 
    Xdata = odrSD.xplus.T # best est. explanatory variables: Cs,Cict,CE,Lict,To
    
    inL = avhrrS.measEq(Xdata, inC) # radiance from input coeffs
    calL = avhrrS.measEq(Xdata, bodr) # calibrated radiance from fit
    # radiance uncertainty from coeffs uncert. evaluated by ODR
    cLU = avhrrS.harUnc(Xdata,bodr,covodr) 
    uX = sqrt(Hrnd[:,1:6]**2 + Hsys[:,1:6]**2) # uncertainty of X variables
    # radiance uncertainty from data and coeffs uncert. evaluated via MC
    mccLU = avhrrS.harUnc(Xdata,bodr,MCcov) # coeffs uncert. evaluated via MC
    mcLU = avhrrS.uncLE(Xdata,bodr,uX,MCcov) # data & coeffs uncertainty

    # graphs of radiance bias with 2sigma error bars
    selMU = sample(range(noMU), nobj) # Select matchups for plotting
    
    plot_ttl = slab + ' Radiance bias and ' + r'$4*\sigma$'+ ' uncertainty from ODR eval. of coeffs\' covariance'
    vis.LbiasU(inL[selMU], calL[selMU], cLU[selMU], 4, plot_ttl) # from ODR evaluation of uncertainty
    plot_ttl = slab + ' Radiance bias and ' + r'$k*\sigma$'+ ' uncertainty from coeffs\' covariance in MC with ' + mclab
    vis.LbiasU(inL[selMU], calL[selMU], mccLU[selMU], k, plot_ttl) # from MC evaluation of uncertainty
    plot_ttl = slab + ' Radiance bias and ' + r'$k*\sigma$'+ ' uncertainty from data and coeffs\' covariance in MC with ' + mclab
    vis.LbiasU(inL[selMU], calL[selMU], mcLU[selMU], k, plot_ttl) # from MC evaluation of uncertainty
    plot_ttl = slab + ' Radiance uncertainty bias from coeffs\' uncertainty evaluated by MC with ' + mclab + ' and by ODR'
    vis.LUdiff(inL[selMU], cLU[selMU], mccLU[selMU], plot_ttl)


""" Graphs of ODR residuals on simulated data againts different variables.
Arguments of the function:
    - avhrrS: instance of class AVHRR series 
    - noMU: number of matchups in the current pair
    - nobj: sample size of select records for plotting
    - slab: label of the sensor being fitted
    - muTime: matchup times
    - Hrnd: matrix of random uncertainties of harmonisation variables
    - odrSD: ODR full output in fitting simulated data
    - weight: switch between weighted residulas or not 
"""
def plotSDfit(avhrrS, noMU, nobj, slab, muTime, Hrnd, odrSD, weight=None):

    # compile data for graphs
    bodr = odrSD.beta # odr fit coefficients
    Xdata = odrSD.xplus.T # best est. explanatory variables: Cs,Cict,CE,Lict,To
    calL = avhrrS.measEq(Xdata, bodr) # calibrated radiance from fit
    Xerr = odrSD.delta.T # true errors for explanatory variables
    notd = avhrrS.notime # time-dependancy: True/False
    
    if weight is not None:
        wlab = ' weighted' # text for plots' title
        
        # evaluate weighted fit residuals
        LEres = odrSD.eps / Hrnd[:,11] # odr estimated error of Y var: K
        
        # weighted true errors of explanatory variables
        if notd:
            cols = [x for x in range(Hrnd.shape[1]) if (x!=5 and x<10)]
        else:
            cols = [x for x in range(Hrnd.shape[1]) if x < 11]
        Xerr = divide(Xerr, Hrnd[:,cols])
     
    selMU = sample(range(noMU), nobj) # Select matchups for plotting
    
    # graphs of residuals, i.e. estimated Y true error, againts different vars
    plot_ttl = slab + wlab + ' residuals from ODR fit'
    vis.resLfit(LEres[selMU],calL[selMU],'Radiance','Fit residuals',plot_ttl)
    sLict = Xdata[selMU, 3] # Lict for the selected matchups
    vis.resLfit(LEres[selMU],sLict,'ICT radiance', 'Fit residuals',plot_ttl)
    sTo = Xdata[selMU, 4] # To (orbit temperature) for the selected matchups
    vis.resLfit(LEres[selMU],sTo,'Orbit temperature (K)','Fit residuals',plot_ttl)
    
    # plot graphs of ODR estimated true error and uncertainties of X vars
    ttl = slab + wlab + ' ODR evaluated true error of '
    plot_ttl = ttl + 'space counts'
    vis.resLfit(Xerr[selMU,0],muTime[selMU],'Time','Counts',plot_ttl)
    plot_ttl = ttl + 'ICT counts'
    vis.resLfit(Xerr[selMU,1],muTime[selMU],'Time','Counts',plot_ttl)
    plot_ttl = ttl + 'Earth counts'
    vis.resLfit(Xerr[selMU,2], muTime[selMU],'Time','Counts',plot_ttl)
    plot_ttl = ttl + 'ICT radiance'
    vis.resLfit(Xerr[selMU,3], muTime[selMU],'Time','Radiance',plot_ttl)
    plot_ttl = ttl + 'orbit temperature'
    vis.resLfit(Xerr[selMU,4], muTime[selMU],'Time','Temperature',plot_ttl)
