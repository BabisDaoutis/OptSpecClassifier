import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

def calc_flux_stats(wavelength, flux, a, b):
    """Calculate the median and standard deviation of the flux of a spectrum over a specified interval.
    Args:
        flux (ndarray): The flux array.
        wave (ndarray): The wavelength array.
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.

    Returns:
        tuple: A tuple containing the median and standard deviation of the flux of the spectrum over the specified interval.
    """
    indices = np.where((wavelength >= a) & (wavelength <= b))[0]
    flux_interval = flux[indices]
    flux_mean = np.mean(flux_interval)
    flux_std = np.std(flux_interval)

    return flux_mean / flux_std

def SN_clean(dfr, band, SN):
    """
    Cleans a DataFrame by removing rows where the signal-to-noise ratio (SNR)
    in a specified spectral region is below a given threshold the SN is calulated from calc_flux_stats()
    """
    dfr.reset_index(inplace=True,drop=True)
    print('Galaxies before cleaning '+band+': ',len(dfr))
    dfr = dfr.loc[(dfr.index[np.where((dfr[band] > SN))].tolist())]
    print('Galaxies after cleaning ' +band+' :',len(dfr))
    return dfr

def calc_eqw_mc(wavelength, flux, sigma, line_center, line_width, cont_width, mc_iter):
    """
    Calculates the Equivalent Width (EW) of a spectral line using Monte Carlo (MC)
    simulations to estimate the value and its uncertainty.

    The function isolates a spectral line, estimates the continuum level from 
    adjacent sidebands, and uses random flux perturbations to determine the 
    robustness of the EW measurement.

    Parameters
    ----------
    wavelength : array_like
        The wavelength array of the spectrum.
    flux : array_like
        The flux array of the spectrum.
    sigma : array_like
        The uncertainty (error) array of the flux. 
        Note: Current logic uses empirical scatter of the continuum for MC noise generation, 
        mostly overriding this input for the simulation steps.
    line_center : float
        The central wavelength of the feature to measure.
    line_width : float
        The full width of the region to consider as the spectral line.
    cont_width : float
        The width of the continuum regions (sidebands) on either side of the line.
    mc_iter : int
        The number of Monte Carlo iterations to perform.

    Returns
    -------
    mean_ew : float
        The mean Equivalent Width derived from the MC iterations.
    std_ew : float
        The standard deviation of the EW distribution (representing the error).
    """
    sigma = np.where(sigma==0, 1, sigma)
    
    line_range = (wavelength > line_center - line_width/2) & (wavelength < line_center + line_width/2)
    cont_range_lf = (wavelength < line_center - line_width/2) & (wavelength > line_center - line_width/2 - cont_width)
    cont_range_rt = (wavelength > line_center + line_width/2) & (wavelength < line_center + line_width/2 + cont_width)

    w_cont_left = wavelength[cont_range_lf]
    w_cont_right = wavelength[cont_range_rt]
    w_cont_tot = np.concatenate((w_cont_left, w_cont_right))

    flux_cont_left = flux[cont_range_lf]
    flux_cont_right = flux[cont_range_rt]
    flux_cont_tot = np.concatenate((flux_cont_left, flux_cont_right))
    
    ew = []
    
    wv_line = wavelength[line_range]
    flux_line = flux[line_range]
    sigma_line = np.std(abs(flux_cont_tot))
    sigma_cont = np.std(abs(flux_cont_tot))
    
    for _ in range(0, mc_iter):
        pert_flux = np.random.normal(loc=flux_line, scale=sigma_line)
        
        wv_interp_line = np.linspace(min(wv_line),max(wv_line),1000)
        cs1 = CubicSpline(wv_line, pert_flux)
        flux_interp_line = cs1(wv_interp_line)

        def func(x, a, b):
            return a * x + b
        
        popt, _ = curve_fit(func, w_cont_tot, np.random.normal(loc=flux_cont_tot, scale=sigma_cont*np.ones(len(flux_cont_tot))), p0=None, sigma=sigma_cont*np.ones(len(flux_cont_tot)))

        flux_cont_line = np.array(popt[0] * wv_interp_line + popt[1])

        ew.append(trapezoid((1 - flux_interp_line/flux_cont_line), wv_interp_line, dx=0.001))
    ew  = np.array(ew)
    return np.mean(ew).item(), np.std(ew).item()

def rf_classify_mc(wavelength, flux, sigma, clf, scaler_sv, plot, n_mc=100):
    """
    Performs Monte Carlo spectral classification using a Random Forest classifier based on 
    emission line Equivalent Widths (EWs).

    This function extracts EWs for H-alpha, [OIII], and H-beta. It accounts for measurement 
    uncertainties by generating `n_mc` synthetic datasets, where feature values are drawn 
    from normal distributions defined by the measured EWs (mean) and their uncertainties (std).
    It then classifies these synthetic points to provide a probabilistic classification.

    Parameters
    ----------
    wavelength : array-like
        The 1D array of spectral wavelengths (typically in Angstroms).
    flux : array-like
        The 1D array of spectral flux density corresponding to the wavelength.
    sigma : array-like
        The 1D array of flux uncertainties (noise/error spectrum).
    clf : sklearn.ensemble.RandomForestClassifier
        The pre-trained Random Forest classifier object.
    scaler_sv : sklearn.preprocessing.StandardScaler
        The fitted scaler object used to normalize features before classification. 
        Must match the scaling applied during the training of `clf`.
    plot : bool
        If True, generates and displays a two-panel plot showing the distribution of 
        predicted classes and the mean probabilities per class. 
        If False, the function returns the string label of the majority class.
    n_mc : int, optional
        The number of Monte Carlo iterations (synthetic samples) to generate. Default is 100.

    Returns
    -------
    str or None
        - If `plot` is False: Returns the label of the most frequently predicted class 
          (e.g., 'SF', 'AGN', 'LINER', 'COMP', 'PAS').
        - If `plot` is True: Displays the plot and returns None.

    Notes
    -----
    - Requires an external function `calc_eqw_mc(wavelength, flux, sigma, center, width, cont_width, mc_iter)` 
      to be available in the scope.
    - Assumes the classifier `clf` outputs integer labels 0-4 mapping to:
      {0: 'SF', 1: 'AGN', 2: 'LINER', 3: 'COMP', 4: 'PAS'}.
    - The specific lines measured are H-alpha (6566 A), [OIII] (5007 A), and H-beta (4864 A).
    """
    lines = [
            {'center': 6566, 'width': 80, 'cont': 10}, # H-alpha
            {'center': 5007, 'width': 30, 'cont': 10}, # [OIII]
            {'center': 4864, 'width': 30, 'cont': 10}  # H-beta
        ]
    
    results = [calc_eqw_mc(wavelength, flux, sigma, l['center'], l['width'], l['cont'], mc_iter=100) for l in lines]
  
    EWs = [res[0] for res in results]
    EWs_unc = [res[1] for res in results]

    df = pd.DataFrame()
    df['EW_NII_HA_NII_NON'] = np.random.normal(loc=EWs[0], scale=EWs_unc[0], size=n_mc)
    df['OIII_5007_EQW_NON'] = np.random.normal(loc=EWs[1], scale=EWs_unc[1], size=n_mc)
    df['H_BETA_EQW_NON'] = np.random.normal(loc=EWs[2], scale=EWs_unc[2], size=n_mc)
    
    features = df[["EW_NII_HA_NII_NON", "OIII_5007_EQW_NON", "H_BETA_EQW_NON"]]
    
    X_scaled = scaler_sv.transform(features)
    
    df['mc_class'] = clf.predict(X_scaled)
    probs = clf.predict_proba(X_scaled)
    
    classes = ['SF', 'AGN', 'LINER', 'COMP', 'PAS']
    
    prob_cols = [f'mc_proba_{c}' for c in classes]
    prob_df = pd.DataFrame(probs, columns=prob_cols)
    df = df.join(prob_df)

    class_mapping = {0: 'SF', 1: 'AGN', 2: 'LINER', 3: 'COMP', 4: 'PAS'}
    class_counts = df['mc_class'].value_counts()
    
    class_counts_ordered = pd.Series({label: class_counts.get(i, 0) for i, label in class_mapping.items()})
 
    mean_probabilities = df[prob_cols].mean(axis=0)
    std_probabilities = df[prob_cols].std(axis=0)

    majority_class = class_counts_ordered.idxmax()

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        ax[0].bar(class_counts_ordered.index, class_counts_ordered.values, color='b', edgecolor='black')
        ax[0].set_title('Distribution of Predicted Classes', fontsize=18)
        ax[0].set_ylabel('Counts', fontsize=18)
        ax[0].tick_params(labelsize=14)

        mean_probabilities.index = classes 
        ax[1].bar(classes, mean_probabilities, yerr=std_probabilities, capsize=5, 
                  alpha=0.7, color='g', ecolor='black')
        ax[1].set_title('Mean Class Probabilities', fontsize=18)
        ax[1].set_ylim(bottom=0, top=1.05)
        ax[1].tick_params(labelsize=14)

        plt.tight_layout()
        plt.show()
        return None
    else:
        return majority_class, mean_probabilities, std_probabilities

def peak_flux(data, window_size):
    """
    Calculates the peak flux of a signal after applying a boxcar smoothing filter.

    This function performs a moving average (convolution) on the input data 
    using a normalized box window, then identifies the maximum value in the 
    resulting smoothed array.

    Args:
        data (array-like): The input signal or data array (1D).
        window_size (int): The width of the smoothing window (number of samples).

    Returns:
        float: The maximum value found in the smoothed data.
    """
    box = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, box, mode='same')
    
    pk_flx = max(smoothed_data)
    return pk_flx


def find_closest_point(data, target_value):
    """
    Finds the element in an array that is numerically closest to a specific target value.

    Args:
        data (array-like): The input array of numbers.
        target_value (float or int): The reference value to search for.

    Returns:
        tuple: A tuple containing:
            - index (int): The index of the closest element.
            - value (float or int): The actual value of the closest element.
    """
    differences = np.abs(data - target_value)
    
    closest_index = np.argmin(differences)
    closest_point = (closest_index, data[closest_index])
    
    return closest_point

def continuum(wavelength, flux, line_center, line_width, cont_width, lm):
    cont_range_lf = (wavelength < line_center - line_width/2) & (wavelength > line_center - line_width/2 - cont_width)
    cont_range_rt = (wavelength > line_center + line_width/2) & (wavelength < line_center + line_width/2 + cont_width)

    cont_left = np.median(flux.loc[cont_range_lf])
    cont_right = np.median(flux.loc[cont_range_rt])

    cont_left = np.median(flux.loc[cont_range_lf])
    cont_right = np.median(flux.loc[cont_range_rt])

    x1, x2 = np.median(wavelength.loc[cont_range_lf]), np.median(wavelength.loc[cont_range_rt])
    y1, y2 = cont_left, cont_right
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    continuum =  m * lm + c
    return continuum

def fwqm(flux, wavelength, line):
    if line == 'ha':
        line_center = 6564
        line_width = 80
        cont_width = 10
        start = 6500
        end = 6650
        window_size = 3
        ln_wv = 6564
        dl = 7
    elif line == 'hb':
        line_center = 4864
        line_width = 30
        cont_width = 10
        start = 4824
        end = 4904
        window_size = 3
        ln_wv = 4864
        dl = 7
    elif line == 'oiii':
        line_center = 5007
        line_width = 30
        cont_width = 10
        start = 4967
        end = 5047
        window_size = 3
        ln_wv = 5007
        dl = 7
    else:
        breakpoint('Invalid line specified. Choose from "ha", "hb", or "oiii".')

    y_range = (wavelength >= ln_wv - dl) & (wavelength <= ln_wv + dl)
    flux_trc_range = (wavelength >= start) & (wavelength <= end)
    wavelength_trc_range = (wavelength >= start) & (wavelength <= end)
    
    y = flux.loc[y_range]
    flux_trc = flux.loc[flux_trc_range]
    wavelength_trc = wavelength.loc[wavelength_trc_range]

    cnt = continuum(wavelength_trc, flux_trc, line_center, line_width, cont_width, line_center)
    pk = peak_flux(y, window_size)
    fwqm = (pk + 3*cnt)/4.
    # plt.plot(wavelength_trc, flux_trc)

    spl1 = PchipInterpolator(wavelength_trc, flux_trc)
    xs = np.linspace(min(wavelength_trc), max(wavelength_trc),1000)
    points = []
    intp = spl1(xs)
    for i in range(0,len(intp)-1):
        if (((intp[i] >= fwqm) & (intp[i+1] <= fwqm)) | ((intp[i+1] >= fwqm) & (intp[i] <= fwqm))):
            points.append(i)
    ha_wv_ind = find_closest_point(xs, line_center)[0]
    l1 = list(filter(lambda x: x < ha_wv_ind, points))
    l2 = list(filter(lambda x: x > ha_wv_ind, points))
    
    if not l1 or not l2:
        l_left = l_right = 0
        return np.abs(xs[l_left]-xs[l_right]).item()
    else:
        l_left = max(list(filter(lambda x: x < ha_wv_ind, points)))
        l_right = min(list(filter(lambda x: x > ha_wv_ind, points)))
        return np.abs(xs[l_left]-xs[l_right]).item()

