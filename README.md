# Harmonisation
FIDUCEO harmonisation modules

There are two main scripts that call the others scripts to perform harmonisation of sensors of a satellite series. The current version of the code works with AVHRR series both simulated and real datasets. The measurement equation coefficients of each sensor in the series are evaluated from matchup data of satellite sensor pairs via an orthogonal distance regression via ODRPACK implementation in SciPy.  

The script seriesH.py performs harmonisation of sensors in the series as ODR regression, assuming there are only random errors present in the calibration data/variables. The covariance matrix of coefficients evaluated by ODR is propagated to Earth radiance uncertainty via the GUM law of uncertainty propagation. 

The script harODRMC adds the full correlation structure of errors via Monte Carlo (MC) computations. It evaluates the calibration coefficients of sensor model in each MC trial via ODR regression on perturbed best estimates of variables. The covariance of coefficients from MC trials is propagated to Earth radiance uncertainty via GUM law of propagation. 
