import pandas as pd
import numpy as np
import xarray as xr
# import netCDF4 # netcdf4 is the engine, not usually imported directly
from scipy.stats import expon, gamma, kendalltau
from scipy.optimize import minimize
from scipy.integrate import quad
import os
import math

# ==============================================================================
# 1. EVENT DETECTION FUNCTION (REFACTORED)
# This function is now designed to work on a single grid point's time series.
# ==============================================================================

def detect_annual_compound_events(tmp_series, spei_series, unique_years,
                                  tmp_threshold, spei_threshold, 
                                  duration_threshold):
    """
    Detects compound events for a single time series and returns annual maxima
    and all individual event characteristics.
    
    Args:
        tmp_series (np.ndarray): 1D NumPy array of temperature data.
        spei_series (np.ndarray): 1D NumPy array of SPEI data.
        unique_years (np.ndarray): Array of unique years for annual analysis.
        tmp_threshold (float): Temperature threshold.
        spei_threshold (float): SPEI threshold.
        duration_threshold (int): Minimum event duration.

    Returns:
        tuple: A tuple containing:
            - grid_annual_max_duration (np.ndarray): Annual max duration for this grid point.
            - grid_annual_max_severity (np.ndarray): Annual max severity for this grid point.
            - all_durations (np.ndarray): Durations of all events at this grid point.
            - all_severities (np.ndarray): Severities of all events at this grid point.
    """
    # Create a DataFrame to easily group by year
    time_index = pd.to_datetime(tmp_series.time.values)
    df = pd.DataFrame({
        'tmp': tmp_series.values,
        'spei': spei_series.values
    }, index=time_index)

    grid_annual_max_duration = np.full(len(unique_years), np.nan)
    grid_annual_max_severity = np.full(len(unique_years), np.nan)
    all_durations = []
    all_severities = []

    for year_idx, year in enumerate(unique_years):
        year_data = df[df.index.year == year]
        if year_data.empty:
            continue

        year_values = year_data.values
        
        events = []
        current_event = []
        
        for i, (tmp_val, spei_val) in enumerate(year_values):
            if tmp_val > tmp_threshold and spei_val < spei_threshold:
                current_event.append((i, tmp_val, spei_val))
            else:
                if len(current_event) >= duration_threshold:
                    events.append(current_event)
                current_event = []
        
        if len(current_event) >= duration_threshold:
            events.append(current_event)
            
        if events:
            durations = [len(event) for event in events]
            severities = [
                sum((tmp_val - tmp_threshold)*math.abs(spei_val) for _, tmp_val, spei_val in event)
                for event in events
            ]
            
            all_durations.extend(durations)
            all_severities.extend(severities)
            
            grid_annual_max_duration[year_idx] = max(durations)
            grid_annual_max_severity[year_idx] = max(severities)/max(durations)

    return grid_annual_max_duration, grid_annual_max_severity, np.array(all_durations), np.array(all_severities)

# ==============================================================================
# 2. COPULA AND RETURN PERIOD FUNCTIONS (CORRECTED)
# ==============================================================================

def fit_marginal_distributions(data, dist_type):
    if len(data) < 10: # Require a minimum number of data points for a stable fit
        return None
    try:
        if dist_type == 'expon':
            loc, scale = expon.fit(data, floc=0) # Fix location to 0 for duration
            return {'loc': loc, 'scale': scale}
        elif dist_type == 'gamma':
            a, loc, scale = gamma.fit(data, floc=0) # Fix location to 0 for severity
            return {'a': a, 'loc': loc, 'scale': scale}
        else:
            raise ValueError("Unsupported distribution type.")
    except Exception: # Catch potential fitting errors from scipy
        return None

def get_cdf_value(data, params, dist_type):
    if params is None: return np.nan
    if dist_type == 'expon':
        return expon.cdf(data, loc=params['loc'], scale=params['scale'])
    elif dist_type == 'gamma':
        return gamma.cdf(data, a=params['a'], loc=params['loc'], scale=params['scale'])
    else:
        raise ValueError("Unsupported distribution type.")

def gumbel_copula_cdf(u, v, theta):
    if theta < 1: return u * v
    u = np.clip(u, 1e-10, 1 - 1e-10)
    v = np.clip(v, 1e-10, 1 - 1e-10)
    return np.exp(-(((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta)))

def fit_gumbel_copula(u_data, v_data):
    if len(u_data) < 2: return 1.0
    tau, _ = kendalltau(u_data, v_data)
    if np.isnan(tau) or tau < 0: return 1.0
    if tau >= 1.0: return np.inf # Handle perfect correlation
    theta = 1 / (1 - tau)
    return max(theta, 1.0)

def calculate_joint_return_period(duration_val, severity_val,
                                  marginal_params_d, marginal_params_s,
                                  dist_type_d, dist_type_s, copula_theta,
                                  lambda_val): # <-- Mean event rate added
    """Calculates the joint return period in years."""
    if np.isnan(duration_val) or np.isnan(severity_val) or lambda_val == 0:
        return np.nan
        
    u_transform = get_cdf_value(duration_val, marginal_params_d, dist_type_d)
    v_transform = get_cdf_value(severity_val, marginal_params_s, dist_type_s)

    if np.isnan(u_transform) or np.isnan(v_transform):
        return np.nan

    # Joint exceedance probability P(D >= d, S >= s) using the survival copula
    # P_exceed = P(U > u, V > v) = C_survival(1-u, 1-v)
    joint_exceedance_prob = gumbel_copula_cdf(1 - u_transform, 1 - v_transform, copula_theta)

    if joint_exceedance_prob == 0:
        return np.inf
    
    # Return Period T = 1 / (lambda * P_exceed)
    return 1 / (lambda_val * joint_exceedance_prob)

# ==============================================================================
# 3. MAIN EXECUTION SCRIPT
# ==============================================================================

# --- Data Loading and Preprocessing ---
# NOTE: Using a hardcoded path. Ensure this file exists.
DATA_PATH = 'I:/temp/CN_max_tmp_1961_2022.nc' 
SPEI_DATA_PATH = 'I:/temp/SPEI_1961_2022.nc' 
try:
    full_dataset = xr.load_dataset(DATA_PATH)
    SPEI_full_dataset = xr.load_dataset(SPEI_DATA_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

# Rename coordinates for consistency if they exist
if 'longitude' in full_dataset.coords:
    full_dataset = full_dataset.rename({'longitude': 'lon'})
if 'latitude' in full_dataset.coords:
    full_dataset = full_dataset.rename({'latitude': 'lat'})
    
tmp_data = full_dataset['t2m'].sel(lat=slice(15, 35), lon=slice(70, 105))
spei_data = SPEI_full_dataset['spei'].sel(lat=slice(15, 35), lon=slice(70, 105))

print(f"Processing data with shape: {tmp_data.shape}")

unique_years = np.unique(tmp_data.time.dt.year.values)
num_grid_cells = len(tmp_data.lat) * len(tmp_data.lon)

# --- Initialize data collectors and storage arrays ---
global_all_event_durations = []
global_all_event_severities = []

# Initialize DataArrays to store results for all years at all grid points
coords = {'year': unique_years, 'lat': tmp_data.lat.values, 'lon': tmp_data.lon.values}
all_years_duration = xr.DataArray(np.nan, dims=['year', 'lat', 'lon'], coords=coords)
all_years_severity = xr.DataArray(np.nan, dims=['year', 'lat', 'lon'], coords=coords)
all_years_return_period = xr.DataArray(np.nan, dims=['year', 'lat', 'lon'], coords=coords)
coords_spatial = {'lat': tmp_data.lat.values, 'lon': tmp_data.lon.values}
all_locations_lambda = xr.DataArray(np.nan, dims=['lat', 'lon'], coords=coords_spatial)
# ==============================================================================
# 4. SINGLE EFFICIENT LOOP FOR DATA PROCESSING
# ==============================================================================

print("\nProcessing all grid points to detect events and find annual maxima...")
# --- Loop ONCE through each grid point ---
for lat_idx, lat_val in enumerate(tmp_data.lat.values):
    for lon_idx, lon_val in enumerate(tmp_data.lon.values):
        print(f"  Processing grid point ({lat_idx+1}/{len(tmp_data.lat)}, {lon_idx+1}/{len(tmp_data.lon)}): Lat={lat_val:.2f}, Lon={lon_val:.2f}")
        
        # Select the time series for the current grid point
        tmp_grid_series = tmp_data.isel(lat=lat_idx, lon=lon_idx)
        spei_grid_series = spei_data.isel(lat=lat_idx, lon=lon_idx)
        
        # Call the refactored detection function ONCE
        grid_ann_max_d, grid_ann_max_s, grid_all_d, grid_all_s = detect_annual_compound_events(
            tmp_series=tmp_grid_series,
            spei_series=spei_grid_series,
            unique_years=unique_years,
            tmp_threshold=305.15,
            spei_threshold=-1.0,
            duration_threshold=3
        )
        
        # Store the annual maxima for this grid point
        all_years_duration[:, lat_idx, lon_idx] = grid_ann_max_d
        all_years_severity[:, lat_idx, lon_idx] = grid_ann_max_s
        num_events_in_grid = len(grid_all_d)
        num_years_in_grid = len(unique_years)
        local_lambda = num_events_in_grid / num_years_in_grid if num_years_in_grid > 0 else 0
        all_locations_lambda[lat_idx, lon_idx] = local_lambda
        # Collect all events from this grid point for global fitting
        if len(grid_all_d) > 0:
            global_all_event_durations.extend(grid_all_d)
            global_all_event_severities.extend(grid_all_s)

# --- Global fitting of marginals and copula ---
print("\nFitting global statistical models...")
global_all_event_durations = np.array(global_all_event_durations)
global_all_event_severities = np.array(global_all_event_severities)

# Calculate mean annual frequency of events (lambda)
# total_years_in_dataset = len(unique_years) * num_grid_cells
# lambda_val = len(global_all_event_durations) / total_years_in_dataset if total_years_in_dataset > 0 else 0
# print(f"Total detected events globally: {len(global_all_event_durations)}")
# print(f"Mean annual event frequency (lambda): {lambda_val:.4f}")

global_marginal_params_duration = fit_marginal_distributions(global_all_event_durations, 'expon')
global_marginal_params_severity = fit_marginal_distributions(global_all_event_severities, 'gamma')

if global_marginal_params_duration is None or global_marginal_params_severity is None:
    print("Could not fit global marginal distributions. Not enough data. Exiting.")
    exit()

print(f"Global Duration marginal (Exponential) params: {global_marginal_params_duration}")
print(f"Global Severity marginal (Gamma) params: {global_marginal_params_severity}")

global_u_data = get_cdf_value(global_all_event_durations, global_marginal_params_duration, 'expon')
global_v_data = get_cdf_value(global_all_event_severities, global_marginal_params_severity, 'gamma')

valid_indices = ~np.isnan(global_u_data) & ~np.isnan(global_v_data)
global_copula_theta = fit_gumbel_copula(global_u_data[valid_indices], global_v_data[valid_indices])
print(f"Fitted Global Gumbel Copula theta: {global_copula_theta:.4f}")

# ==============================================================================
# 5. CALCULATE RETURN PERIODS AND SAVE RESULTS
# ==============================================================================

print("\nCalculating return periods for all annual maximum events...")
for lat_idx in range(len(tmp_data.lat)):
    for lon_idx in range(len(tmp_data.lon)):
        lambda_for_grid = all_locations_lambda.values[lat_idx, lon_idx]
        if lambda_for_grid==0:
            continue
        for year_idx in range(len(unique_years)):

            
            duration_val = all_years_duration.values[year_idx, lat_idx, lon_idx]
            severity_val = all_years_severity.values[year_idx, lat_idx, lon_idx]
            if np.isnan(duration_val) or np.isnan(severity_val):
                continue
                
            rp = calculate_joint_return_period(
                duration_val=duration_val,
                severity_val=severity_val,
                marginal_params_d=global_marginal_params_duration,
                marginal_params_s=global_marginal_params_severity,
                dist_type_d='expon',
                dist_type_s='gamma',
                copula_theta=global_copula_theta,
                lambda_val=lambda_for_grid # Use Use the specific lambda for this grid
            )
            all_years_return_period[year_idx, lat_idx, lon_idx] = rp

# --- Save results to annual NetCDF files ---
output_dir = 'compound_event_return_periods_netcdf'
os.makedirs(output_dir, exist_ok=True)
print(f"\nSaving results to annual NetCDF files in '{output_dir}/'...")

for year_idx, year in enumerate(unique_years):
    annual_ds = xr.Dataset({
        'duration': all_years_duration.isel(year=year_idx),
        'severity': all_years_severity.isel(year=year_idx),
        'return_period': all_years_return_period.isel(year=year_idx)
    })
    
    annual_ds.duration.attrs = {'long_name': 'Annual Max Compound Event Duration', 'units': 'days'}
    annual_ds.severity.attrs = {'long_name': 'Annual Max Compound Event Severity', 'units': 'cumulative_euclidean_distance'}
    annual_ds.return_period.attrs = {'long_name': 'Joint Return Period of Annual Max Event', 'units': 'years'}

    output_filename = os.path.join(output_dir, f'compound_event_metrics_{year}.nc')
    annual_ds.to_netcdf(output_filename, engine="netcdf4")
    print(f"  Saved {output_filename}")

print("\nProcessing complete.")

# import pandas as pd
# import numpy as np
# import xarray as xr
# import netcdf4
# from scipy.stats import expon, gamma, kendalltau
# from scipy.optimize import minimize
# from scipy.integrate import quad
# import os # Import os for path manipulation
# import math

# def detect_annual_compound_events(tmp_data, spei_data, 
                                  # tmp_threshold, spei_threshold, 
                                  # duration_threshold):
    # """
    # Detects annual compound heat-drought events from temperature and SPEI data.

    # A compound event is defined as a period where temperature is above its 
    # threshold and SPEI is below its threshold simultaneously for a minimum duration.

    # Args:
        # tmp_data (xr.DataArray): Time series of temperature data (time, latitude, longitude).
        # spei_data (xr.DataArray): Time series of SPEI data (time, latitude, longitude).
        # tmp_threshold (float): The temperature threshold for defining a heat event.
        # spei_threshold (float): The SPEI threshold for defining a drought event.
        # duration_threshold (int): The minimum number of consecutive time steps 
                                  # to qualify as an event.

    # Returns:
        # tuple: A tuple containing:
            # - annual_stats (xr.Dataset): A dataset with the annual maximum duration 
              # and annual maximum severity for each grid cell.
            # - all_event_durations (np.ndarray): A 1D array of all detected event durations.
            # - all_event_severities (np.ndarray): A 1D array of all detected event severities.
    # """
    # # Ensure time coordinates are in datetime format
    # tmp_data = tmp_data.assign_coords(time=pd.to_datetime(tmp_data.time.values))
    # spei_data = spei_data.assign_coords(time=pd.to_datetime(spei_data.time.values))

    # years = tmp_data.time.dt.year.values
    # unique_years = np.unique(years)

    # # Lists to store characteristics of all detected events across all grid points
    # all_event_durations = []
    # all_event_severities = []

    # # Prepare the coordinates for the output xarray Dataset
    # coords = {
        # 'year': unique_years,
        # 'lat': tmp_data.lat,
        # 'lon': tmp_data.lon
    # }
    
    # # Initialize an xarray Dataset to store the annual maximum statistics
    # annual_stats = xr.Dataset({
        # var: xr.DataArray(
            # np.full((len(unique_years), len(tmp_data.lat), len(tmp_data.lon)), np.nan, dtype=float),
            # coords=coords, 
            # dims=['year', 'lat', 'lon']
        # ) for var in ['maximum_duration', 'maximum_severity']
    # })

    # # Iterate over each year, latitude, and longitude to process the time series
    # for year_idx, year in enumerate(unique_years):
        # tmp_year_data = tmp_data.sel(time=tmp_data.time.dt.year == year)
        # spei_year_data = spei_data.sel(time=spei_data.time.dt.year == year)

        # for lat_idx in range(len(tmp_data.lat)):
            # for lon_idx in range(len(tmp_data.lon)):
                # tmp_grid_data = tmp_year_data[:, lat_idx, lon_idx].values
                # spei_grid_data = spei_year_data[:, lat_idx, lon_idx].values
                # events = []
                # current_event = []
                # # Identify consecutive days that meet the compound event criteria
                # for i, (tmp_val, spei_val) in enumerate(zip(tmp_grid_data, spei_grid_data)):
                    # if tmp_val > tmp_threshold and spei_val < spei_threshold:
                        # current_event.append((i, tmp_val, spei_val))
                    # else:
                        # if len(current_event) >= duration_threshold:
                            # events.append(current_event)
                        # current_event = []
                
                # # Ensure the last event in the series is captured if it meets the duration
                # if len(current_event) >= duration_threshold:
                    # events.append(current_event)

                # if events:
                    # # Calculate duration and severity for each detected event
                    # durations = [len(event) for event in events]
                    
                    # # Severity is the sum of the daily Euclidean distances from the thresholds
                    # severities = [
                        # sum(math.sqrt((tmp_val - tmp_threshold)**2 + (spei_val - spei_threshold)**2) for _, tmp_val, spei_val in event)
                        # for event in events
                    # ]

                    # # Add the characteristics of events from this grid cell to the global lists
                    # all_event_durations.extend(durations)
                    # all_event_severities.extend(severities)

                    # # Store the annual maximums for the current grid cell
                    # annual_stats['maximum_duration'][year_idx, lat_idx, lon_idx] = max(durations)
                    # annual_stats['maximum_severity'][year_idx, lat_idx, lon_idx] = max(severities)/max(durations)
    # return annual_stats, np.array(all_event_durations), np.array(all_event_severities)

# # --- Copula Functions ---

# def fit_marginal_distributions(data, dist_type):
    # """
    # Fits a specified distribution to the data and returns the fitted parameters.
    # dist_type can be 'expon' or 'gamma'.
    # """
    # if len(data) == 0:
        # return None # Indicate no data to fit
    # if dist_type == 'expon':
        # loc, scale = expon.fit(data)
        # return {'loc': loc, 'scale': scale}
    # elif dist_type == 'gamma':
        # a, loc, scale = gamma.fit(data)
        # return {'a': a, 'loc': loc, 'scale': scale}
    # else:
        # raise ValueError("Unsupported distribution type. Choose 'expon' or 'gamma'.")

# def get_cdf_value(data, params, dist_type):
    # """
    # Returns the CDF value for given data and distribution parameters.
    # """
    # if params is None: # Handle cases where fitting failed
        # return np.nan
        
    # if dist_type == 'expon':
        # return expon.cdf(data, loc=params['loc'], scale=params['scale'])
    # elif dist_type == 'gamma':
        # return gamma.cdf(data, a=params['a'], loc=params['loc'], scale=params['scale'])
    # else:
        # raise ValueError("Unsupported distribution type.")

# def gumbel_copula_cdf(u, v, theta):
    # """Gumbel Copula CDF"""
    # # Handle cases where u or v are extremely close to 0 or 1 to prevent log errors
    # u = np.clip(u, 1e-10, 1 - 1e-10)
    # v = np.clip(v, 1e-10, 1 - 1e-10)
    
    # if theta < 1: # Gumbel copula requires theta >= 1
        # return u * v # Treat as independence if theta is invalid for Gumbel

    # try:
        # term1 = (-np.log(u))**theta
        # term2 = (-np.log(v))**theta
        # result = np.exp(-((term1 + term2)**(1/theta)))
    # except (OverflowError, ValueError, ZeroDivisionError): 
        # # Handle cases where intermediate calculations might lead to errors
        # # This can happen with very small u/v or very large theta
        # result = 0.0 # Assign a sensible default or np.nan depending on desired behavior
    # return result

# def fit_gumbel_copula(u_data, v_data):
    # """
    # Fits the Gumbel Copula parameter theta using Kendall's Tau.
    # For Gumbel Copula, tau = (theta - 1) / theta
    # So, theta = 1 / (1 - tau)
    # """
    # if len(u_data) < 2: # Need at least 2 data points for Kendall's tau
        # return 1.0 # Default to independence if not enough data
        
    # tau, _ = kendalltau(u_data, v_data)
    # if np.isnan(tau): # Handle cases where tau might be NaN
        # return 1.0 # Default to independence
        
    # if tau == 1.0: # Handle perfect correlation case to avoid division by zero
        # theta = np.inf
    # elif tau < 0: # Gumbel copula only for positive dependence. If tau is negative, assume independence (theta=1)
        # theta = 1.0 
    # else:
        # theta = 1 / (1 - tau)
    
    # # Ensure theta is at least 1, as per Gumbel Copula definition
    # return max(theta, 1.0)


# def calculate_joint_return_period(duration_threshold_event, severity_threshold_event,
                                   # marginal_params_d, marginal_params_s,
                                   # dist_type_d, dist_type_s, copula_theta):
    # """
    # Calculates the joint return period of an event (duration >= threshold_d AND severity >= threshold_s).
    
    # Args:
        # duration_threshold_event (float): The specific duration threshold for which to calculate the return period.
        # severity_threshold_event (float): The specific severity threshold for which to calculate the return period.
        # marginal_params_d (dict): Parameters of the fitted marginal distribution for duration.
        # marginal_params_s (dict): Parameters of the fitted marginal distribution for severity.
        # dist_type_d (str): Type of marginal distribution for duration ('expon' or 'gamma').
        # dist_type_s (str): Type of marginal distribution for severity ('expon' or 'gamma').
        # copula_theta (float): The fitted parameter for the Gumbel Copula.
        
    # Returns:
        # float: The joint return period in years.
    # """
    
    # # Check for NaN inputs for thresholds
    # if np.isnan(duration_threshold_event) or np.isnan(severity_threshold_event):
        # return np.nan
        
    # # Transform to uniform [0,1] domain for Copula (u = F_D(d), v = F_S(s))
    # u_transform = get_cdf_value(duration_threshold_event, marginal_params_d, dist_type_d)
    # v_transform = get_cdf_value(severity_threshold_event, marginal_params_s, dist_type_s)

    # # If CDF values are NaN (e.g., if marginal params are None), return NaN
    # if np.isnan(u_transform) or np.isnan(v_transform):
        # return np.nan

    # # Joint exceedance probability P(D >= d, S >= s) using the survival copula
    # u_prime = 1 - u_transform
    # v_prime = 1 - v_transform

    # # Ensure u_prime and v_prime are within (0,1) for log calculations
    # u_prime = np.clip(u_prime, 1e-10, 1 - 1e-10)
    # v_prime = np.clip(v_prime, 1e-10, 1 - 1e-10)

    # joint_exceedance_prob = gumbel_copula_cdf(u_prime, v_prime, copula_theta)

    # if joint_exceedance_prob == 0:
        # return np.inf 
    # return 1 / joint_exceedance_prob


# # --- Main Execution ---

# # Data loading and preprocessing
# tmp_data = xr.load_dataset('I:\temp\CN_max_tmp_1961_2022.nc')['t2m']
# tmp_data = tmp_data.rename({'longitude': 'lon'})
# tmp_data = tmp_data.rename({'latitude': 'lat'})
# CN_data = tmp_data.sel(lat=slice(15,35), lon=slice(70,105))

# spei_data = xr.load_dataset('I:\temp\CN_max_tmp_1961_2022.nc')['spei'] ##[1\2\3\4]

# print(f"Full dataset shape: {CN_data.shape}")

# # Get unique years from the data
# unique_years = np.unique(CN_data.time.dt.year.values)

# # --- Global fitting of marginals and copula (using all events from all grid points) ---
# # This is crucial for a consistent return period calculation across the domain.
# print("\nCollecting all drought events across all grid points for global fitting...")
# global_all_event_durations = []
# global_all_event_severities = []

# # Iterate through all grid points to collect all detected drought events
# for lat_val in CN_data.lat.values:
    # for lon_val in CN_data.lon.values:
        # # Select data for the current grid point (and ensure it's a 1D time series)
        # grid_point_data = CN_data.sel(lat=lat_val, lon=lon_val)
        # spei_point_data = spei_data.sel(lat=lat_val, lon=lon_val)
        
        # # Call detect_annual_events for this single grid point
        # _, grid_all_event_durations, grid_all_event_severities = detect_annual_compound_events(
            # tmp_data=grid_point_data,
            # spei_data = spei_point_data,
            # tmp_threshold = 305.15,
            # spei_threshold=-1.0,
            # duration_threshold=3
        # )
        # global_all_event_durations.extend(grid_all_event_durations)
        # global_all_event_severities.extend(grid_all_event_severities)

# global_all_event_durations = np.array(global_all_event_durations)
# global_all_event_severities = np.array(global_all_event_severities)

# # Filter out NaN values from global lists
# global_all_event_durations = global_all_event_durations[~np.isnan(global_all_event_durations)]
# global_all_event_severities = global_all_event_severities[~np.isnan(global_all_event_severities)]

# print(f"Total detected events globally: {len(global_all_event_durations)}")

# # Fit marginal distributions for duration and severity using ALL detected events globally
# print("\nFitting global marginal distributions...")
# if len(global_all_event_durations) > 0:
    # global_marginal_params_duration = fit_marginal_distributions(global_all_event_durations, 'expon')
    # print(f"Global Duration marginal (Exponential) parameters: {global_marginal_params_duration}")
# else:
    # print("Not enough global duration data to fit marginal distribution. Exiting.")
    # exit()

# if len(global_all_event_severities) > 0:
    # global_marginal_params_severity = fit_marginal_distributions(global_all_event_severities, 'gamma')
    # print(f"Global Severity marginal (Gamma) parameters: {global_marginal_params_severity}")
# else:
    # print("Not enough global severity data to fit marginal distribution. Exiting.")
    # exit()

# # Transform global data to uniform [0,1] domain for copula fitting
# global_u_data = get_cdf_value(global_all_event_durations, global_marginal_params_duration, 'expon')
# global_v_data = get_cdf_value(global_all_event_severities, global_marginal_params_severity, 'gamma')

# # Filter out NaNs from transformed data before copula fitting
# valid_indices_copula = ~np.isnan(global_u_data) & ~np.isnan(global_v_data)
# global_u_data = global_u_data[valid_indices_copula]
# global_v_data = global_v_data[valid_indices_copula]

# # Fit Gumbel Copula parameter theta globally
# print("\nFitting global Gumbel Copula...")
# global_copula_theta = fit_gumbel_copula(global_u_data, global_v_data)
# print(f"Fitted Global Gumbel Copula theta: {global_copula_theta}")

# # Create output directory
# output_dir = 'drought_return_periods_netcdf'
# os.makedirs(output_dir, exist_ok=True)

# # --- Iterate through each year and each grid point to calculate and save results ---
# print("\nCalculating annual maximum drought event return periods for all grid points...")

# # Initialize DataArrays to store results for all years at all grid points
# # These will be filled inside the loops and then sliced by year for saving.
# all_years_duration = xr.DataArray(
    # np.full((len(unique_years), len(CN_data.lat), len(CN_data.lon)), np.nan),
    # dims=['year', 'lat', 'lon'],
    # coords={'year': unique_years, 'lat': CN_data.lat.values, 'lon': CN_data.lon.values}
# )
# all_years_severity = xr.DataArray(
    # np.full((len(unique_years), len(CN_data.lat), len(CN_data.lon)), np.nan),
    # dims=['year', 'lat', 'lon'],
    # coords={'year': unique_years, 'lat': CN_data.lat.values, 'lon': CN_data.lon.values}
# )
# all_years_return_period = xr.DataArray(
    # np.full((len(unique_years), len(CN_data.lat), len(CN_data.lon)), np.nan),
    # dims=['year', 'lat', 'lon'],
    # coords={'year': unique_years, 'lat': CN_data.lat.values, 'lon': CN_data.lon.values}
# )

# # Loop through each grid point
# for lat_idx, lat_val in enumerate(CN_data.lat.values):
    # for lon_idx, lon_val in enumerate(CN_data.lon.values):
        # print(f"  Processing grid point: Lat={lat_val:.2f}, Lon={lon_val:.2f}")
        
        # # Select time series data for the current grid point
        # grid_point_data = CN_data.sel(lat=lat_val, lon=lon_val)
        
        # # Detect annual maximum events for this specific grid point
        # # The detect_annual_events function now returns max_duration and max_severity for a single grid point
        # grid_annual_max_duration, grid_annual_max_severity, _, _ = detect_annual_events(
            # data=grid_point_data,
            # threshold=305.15,
            # duration_threshold=3
        # )
        
        # # Store the detected annual maxima into the global arrays
        # all_years_duration[:, lat_idx, lon_idx] = grid_annual_max_duration
        # all_years_severity[:, lat_idx, lon_idx] = grid_annual_max_severity

        # # Calculate return period for each year's maximum event at this grid point
        # for year_idx, year in enumerate(unique_years):
            # duration_val = grid_annual_max_duration[year_idx]
            # severity_val = grid_annual_max_severity[year_idx]

            # if not (np.isnan(duration_val) or np.isnan(severity_val)):
                # return_period = calculate_joint_return_period(
                    # duration_threshold_event=duration_val,
                    # severity_threshold_event=severity_val,
                    # marginal_params_d=global_marginal_params_duration, # Use globally fitted parameters
                    # marginal_params_s=global_marginal_params_severity, # Use globally fitted parameters
                    # dist_type_d='expon',
                    # dist_type_s='gamma',
                    # copula_theta=global_copula_theta # Use globally fitted parameter
                # )
                # all_years_return_period[year_idx, lat_idx, lon_idx] = return_period
            # # else: If NaN, it remains NaN from initialization

# # --- Save results to annual NetCDF files ---
# print("\nSaving results to annual NetCDF files...")

# for year_idx, year in enumerate(unique_years):
    # # Create an xarray.Dataset for the current year
    # annual_ds = xr.Dataset({
        # 'duration': all_years_duration.isel(year=year_idx).drop_vars('year'),
        # 'severity': all_years_severity.isel(year=year_idx).drop_vars('year'),
        # 'return_period': all_years_return_period.isel(year=year_idx).drop_vars('year')
    # })
    
    # # Add attributes for clarity
    # annual_ds['duration'].attrs['long_name'] = 'Annual Maximum Drought Duration'
    # annual_ds['duration'].attrs['units'] = 'months'
    # annual_ds['severity'].attrs['long_name'] = 'Annual Maximum Drought Severity'
    # annual_ds['severity'].attrs['units'] = 'SPEI_sum'
    # annual_ds['return_period'].attrs['long_name'] = 'Joint Return Period of Annual Maximum Drought Event'
    # annual_ds['return_period'].attrs['units'] = 'years'

    # # Define output file path
    # output_filename = os.path.join(output_dir, f'drought_metrics_{year}.nc')
    
    # # Save the Dataset to NetCDF
    # annual_ds.to_netcdf(output_filename)
    # print(f"  Saved {output_filename}")

# print("\nProcessing complete. All annual NetCDF files generated.")

# # import pandas as pd
# # import numpy as np
# # import xarray as xr
# # from scipy.stats import expon, gamma, kendalltau
# # from scipy.optimize import minimize
# # from scipy.integrate import quad
# # from statsmodels.distributions.empirical_distribution import ECDF # For empirical CDF

# # # (Your existing detect_annual_events function goes here)
# # def detect_annual_events(data, threshold, duration_threshold):
    # # # Ensure time coordinate is in datetime format
    # # data = data.assign_coords(time=pd.to_datetime(data.time.values))
    # # years = data.time.dt.year.values
    # # unique_years = np.unique(years)
    
    # # # Get spatial dimension sizes (handle scalar case)
    # # n_lat = 1 if data.lat.ndim == 0 else len(data.lat)
    # # n_lon = 1 if data.lon.ndim == 0 else len(data.lon)

    # # # Create coordinate system
    # # coords = {
        # # 'year': unique_years,
        # # 'lat': [data.lat.values] if data.lat.ndim == 0 else data.lat.values,
        # # 'lon': [data.lon.values] if data.lon.ndim == 0 else data.lon.values
    # # }

    # # # Create empty datasets for annual max duration and severity
    # # annual_stats = xr.Dataset({
        # # 'max_duration': xr.DataArray(
            # # np.full((len(unique_years), n_lat, n_lon), np.nan),  # Initialize with NaN
            # # dims=['year', 'lat', 'lon'], 
            # # coords=coords
        # # ),
        # # 'max_severity': xr.DataArray(
            # # np.full((len(unique_years), n_lat, n_lon), np.nan),  # Initialize with NaN
            # # dims=['year', 'lat', 'lon'], 
            # # coords=coords
        # # )
    # # })

    # # # Lists to store all detected event durations and severities for Copula analysis
    # # all_event_durations = []
    # # all_event_severities = []
        
    # # for year_idx, year in enumerate(unique_years):
        # # year_data = data.sel(time=data.time.dt.year == year)
        
        # # for lat_idx in range(n_lat):
            # # for lon_idx in range(n_lon):
                # # # Get time series for current grid point
                # # if n_lat > 1 and n_lon > 1:
                    # # grid_data = year_data.isel(lat=lat_idx, lon=lon_idx).values
                # # elif n_lat > 1:
                    # # grid_data = year_data.isel(lat=lat_idx).values
                # # elif n_lon > 1:
                    # # grid_data = year_data.isel(lon=lon_idx).values
                # # else:
                    # # # Scalar case, get value directly
                    # # grid_data = year_data.values
                
                # # # Ensure data is a 1D array (time series)
                # # if grid_data.ndim > 1:
                    # # # Try to flatten multi-dimensional array to 1D
                    # # grid_data = grid_data.flatten()
                
                # # # Detect drought events
                # # events = []
                # # current_event = []
                
                # # for i, value in enumerate(grid_data):
                    # # if value < threshold:
                        # # current_event.append((i, value))
                    # # else:
                        # # if len(current_event) >= duration_threshold:
                            # # events.append(current_event)
                        # # current_event = []
                
                # # # Check for the last event
                # # if len(current_event) >= duration_threshold:
                    # # events.append(current_event)
                
                # # # Calculate and store statistics
                # # if events:
                    # # durations = [len(event) for event in events]
                    # # severities = [sum(-val for _, val in event) for event in events] # Severity as positive value (deficit)

                    # # # Store for Copula analysis (for all events, not just annual max)
                    # # all_event_durations.extend(durations)
                    # # all_event_severities.extend(severities)

                    # # annual_stats['max_duration'][year_idx, lat_idx, lon_idx] = max(durations)
                    # # annual_stats['max_severity'][year_idx, lat_idx, lon_idx] = max(severities)
    
    # # return annual_stats, np.array(all_event_durations), np.array(all_event_severities)

# # # --- Copula Functions ---

# # def fit_marginal_distributions(data, dist_type):
    # # """
    # # Fits a specified distribution to the data and returns the fitted parameters.
    # # dist_type can be 'expon' or 'gamma'.
    # # """
    # # if dist_type == 'expon':
        # # loc, scale = expon.fit(data)
        # # return {'loc': loc, 'scale': scale}
    # # elif dist_type == 'gamma':
        # # a, loc, scale = gamma.fit(data)
        # # return {'a': a, 'loc': loc, 'scale': scale}
    # # else:
        # # raise ValueError("Unsupported distribution type. Choose 'expon' or 'gamma'.")

# # def get_cdf_value(data, params, dist_type):
    # # """
    # # Returns the CDF value for given data and distribution parameters.
    # # """
    # # if dist_type == 'expon':
        # # return expon.cdf(data, loc=params['loc'], scale=params['scale'])
    # # elif dist_type == 'gamma':
        # # return gamma.cdf(data, a=params['a'], loc=params['loc'], scale=params['scale'])
    # # else:
        # # raise ValueError("Unsupported distribution type.")

# # def gumbel_copula_cdf(u, v, theta):
    # # """Gumbel Copula CDF"""
    # # # Handle cases where u or v are extremely close to 0 or 1 to prevent log errors
    # # u = np.clip(u, 1e-10, 1 - 1e-10)
    # # v = np.clip(v, 1e-10, 1 - 1e-10)
    
    # # if theta < 1: # Gumbel copula requires theta >= 1
        # # # For values near 1, it implies independence or negative dependence,
        # # # where Gumbel is not suitable. Handle as independence or return inf for error.
        # # return u * v # Independence case if theta is somehow < 1
    
    # # # Avoid log(0) issues by clipping arguments to log
    # # term1 = (-np.log(u))**theta
    # # term2 = (-np.log(v))**theta
    
    # # # Ensure terms are non-negative and finite before exponentiation
    # # if np.isinf(term1) or np.isinf(term2):
        # # return 0.0 # Or handle as appropriate for extreme values
    
    # # try:
        # # result = np.exp(-((term1 + term2)**(1/theta)))
    # # except OverflowError: # Catch cases where exponent is too large
        # # result = 0.0
    # # return result


# # def fit_gumbel_copula(u_data, v_data):
    # # """
    # # Fits the Gumbel Copula parameter theta using Kendall's Tau.
    # # For Gumbel Copula, tau = (theta - 1) / theta
    # # So, theta = 1 / (1 - tau)
    # # """
    # # tau, _ = kendalltau(u_data, v_data)
    # # if tau == 1.0: # Handle perfect correlation case to avoid division by zero
        # # theta = np.inf
    # # elif tau < 0: # Gumbel copula only for positive dependence. If tau is negative, assume independence (theta=1)
        # # theta = 1.0 # Or raise a warning/error if Gumbel is strictly for positive dependence
    # # else:
        # # theta = 1 / (1 - tau)
    
    # # # Ensure theta is at least 1, as per Gumbel Copula definition
    # # return max(theta, 1.0)


# # def calculate_joint_return_period(duration_threshold_event, severity_threshold_event,
                                   # # marginal_params_d, marginal_params_s,
                                   # # dist_type_d, dist_type_s, copula_theta):
    # # """
    # # Calculates the joint return period of an event (duration >= threshold_d AND severity >= threshold_s).
    
    # # Args:
        # # duration_threshold_event (float): The specific duration threshold for which to calculate the return period.
        # # severity_threshold_event (float): The specific severity threshold for which to calculate the return period.
        # # marginal_params_d (dict): Parameters of the fitted marginal distribution for duration.
        # # marginal_params_s (dict): Parameters of the fitted marginal distribution for severity.
        # # dist_type_d (str): Type of marginal distribution for duration ('expon' or 'gamma').
        # # dist_type_s (str): Type of marginal distribution for severity ('expon' or 'gamma').
        # # copula_theta (float): The fitted parameter for the Gumbel Copula.
        
    # # Returns:
        # # float: The joint return period in years.
    # # """
    
    # # # Transform to uniform [0,1] domain for Copula (u = F_D(d), v = F_S(s))
    # # # We need 1 - F(x) for the survival copula
    # # u_transform = get_cdf_value(duration_threshold_event, marginal_params_d, dist_type_d)
    # # v_transform = get_cdf_value(severity_threshold_event, marginal_params_s, dist_type_s)

    # # # Joint exceedance probability P(D >= d, S >= s) using the survival copula
    # # # For Gumbel, C_survival(u', v') is the same form as C(u', v') where u' = 1-u and v' = 1-v
    # # u_prime = 1 - u_transform
    # # v_prime = 1 - v_transform

    # # # Ensure u_prime and v_prime are within (0,1) for log calculations
    # # u_prime = np.clip(u_prime, 1e-10, 1 - 1e-10)
    # # v_prime = np.clip(v_prime, 1e-10, 1 - 1e-10)

    # # joint_exceedance_prob = gumbel_copula_cdf(u_prime, v_prime, copula_theta)

    # # # Return Period T = 1 / P(event)
    # # # If joint_exceedance_prob is extremely small (close to 0), the return period will be very large.
    # # if joint_exceedance_prob == 0:
        # # return np.inf # Return infinite if probability is zero (event never occurs)
    # # return 1 / joint_exceedance_prob


# # # --- Main Execution ---

# # # Data loading and preprocessing
# # spi_data = xr.load_dataset('E:\chorm_download\spei03.nc')
# # CN_data = spi_data.sel(lat=slice(15,55), lon=slice(70,140))
# # start_date = '1960-01-01'
# # end_date = '2020-12-31'
# # CN_data = CN_data.sel(time=slice(start_date, end_date))
# # sample_data = CN_data['spei'].sel(lon=114.25, lat=30.75, method='nearest').expand_dims(['lat','lon'])
# # print(f"Sample data shape: {sample_data.shape}")

# # # Detect drought events and get all event durations and severities
# # # The first return value `annual_stats` contains the annual maximum duration and severity
# # annual_stats, all_event_durations, all_event_severities = detect_annual_events(
    # # data=sample_data,
    # # threshold=-1.0,
    # # duration_threshold=1
# # )

# # # Filter out NaN values from durations and severities
# # all_event_durations = all_event_durations[~np.isnan(all_event_durations)]
# # all_event_severities = all_event_severities[~np.isnan(all_event_severities)]

# # print(f"\nTotal detected events: {len(all_event_durations)}")
# # if len(all_event_durations) == 0:
    # # print("No events detected. Cannot calculate return period.")
    # # exit()

# # # Fit marginal distributions for duration and severity using ALL detected events
# # print("\nFitting marginal distributions...")
# # # Ensure there's enough data for fitting
# # if len(all_event_durations) > 0:
    # # marginal_params_duration = fit_marginal_distributions(all_event_durations, 'expon')
    # # print(f"Duration marginal (Exponential) parameters: {marginal_params_duration}")
# # else:
    # # print("Not enough duration data to fit marginal distribution.")
    # # exit()

# # if len(all_event_severities) > 0:
    # # marginal_params_severity = fit_marginal_distributions(all_event_severities, 'gamma')
    # # print(f"Severity marginal (Gamma) parameters: {marginal_params_severity}")
# # else:
    # # print("Not enough severity data to fit marginal distribution.")
    # # exit()

# # # Transform data to uniform [0,1] domain using fitted parametric CDFs for copula fitting
# # u_data = get_cdf_value(all_event_durations, marginal_params_duration, 'expon')
# # v_data = get_cdf_value(all_event_severities, marginal_params_severity, 'gamma')

# # # Fit Gumbel Copula parameter theta
# # print("\nFitting Gumbel Copula...")
# # copula_theta = fit_gumbel_copula(u_data, v_data)
# # print(f"Fitted Gumbel Copula theta: {copula_theta}")


# # # --- Calculate Return Periods for Annual Maximum Drought Events and Save to CSV ---
# # print("\nCalculating Return Periods for Annual Maximum Drought Events:")

# # # Prepare a list to store results for CSV
# # results_for_csv = []

# # # Loop through each year in annual_stats to get the annual max duration and severity
# # for year in annual_stats.year.values:
    # # # Ensure to extract scalar values if lat/lon dimensions exist
    # # duration_val = annual_stats['max_duration'].sel(year=year).item()
    # # severity_val = annual_stats['max_severity'].sel(year=year).item()

    # # # Only calculate if valid duration and severity are present for the year
    # # if not (np.isnan(duration_val) or np.isnan(severity_val)):
        # # return_period = calculate_joint_return_period(
            # # duration_threshold_event=duration_val,
            # # severity_threshold_event=severity_val,
            # # marginal_params_d=marginal_params_duration,
            # # marginal_params_s=marginal_params_severity,
            # # dist_type_d='expon',
            # # dist_type_s='gamma',
            # # copula_theta=copula_theta
        # # )
        # # print(f"  {year}年: Max Duration={duration_val:.0f}, Max Severity={severity_val:.2f}, Return Period={return_period:.2f} years")
        # # results_for_csv.append({
            # # 'year': int(year),
            # # 'duration': duration_val,
            # # 'severity': severity_val,
            # # 'return_period': return_period
        # # })
    # # else:
        # # print(f"  {year}年: No valid annual maximum event detected (Duration={duration_val}, Severity={severity_val}). Skipping return period calculation.")
        # # results_for_csv.append({
            # # 'year': int(year),
            # # 'duration': np.nan,
            # # 'severity': np.nan,
            # # 'return_period': np.nan
        # # })

# # # Create DataFrame and save to CSV
# # df_results = pd.DataFrame(results_for_csv)
# # output_csv_path = 'annual_drought_return_periods.csv'
# # df_results.to_csv(output_csv_path, index=False)

# # print(f"\nResults saved to {output_csv_path}")
# # import pandas as pd
# # import numpy as np
# # import xarray as xr
# # from scipy.stats import expon, gamma, kendalltau
# # from scipy.optimize import minimize
# # from scipy.integrate import quad

# # def detect_annual_events(data, threshold, duration_threshold):
    # # # 确保时间坐标是正确的datetime格式
    # # data = data.assign_coords(time=pd.to_datetime(data.time.values))
    # # years = data.time.dt.year.values
    # # unique_years = np.unique(years)
    
    # # # 获取空间维度大小（处理标量情况）
    # # n_lat = 1 if data.lat.ndim == 0 else len(data.lat)
    # # n_lon = 1 if data.lon.ndim == 0 else len(data.lon)

    # # # 创建坐标系统
    # # coords = {
        # # 'year': unique_years,
        # # 'lat': [data.lat.values] if data.lat.ndim == 0 else data.lat.values,
        # # 'lon': [data.lon.values] if data.lon.ndim == 0 else data.lon.values
    # # }

    # # # 创建空数据集
    # # annual_stats = xr.Dataset({
        # # 'max_duration': xr.DataArray(
            # # np.full((len(unique_years), n_lat, n_lon), np.nan),  # 使用NaN初始化
            # # dims=['year', 'lat', 'lon'], 
            # # coords=coords
        # # ),
        # # 'max_severity': xr.DataArray(
            # # np.full((len(unique_years), n_lat, n_lon), np.nan),  # 使用NaN初始化
            # # dims=['year', 'lat', 'lon'], 
            # # coords=coords
        # # )
    # # })
        
    # # for year_idx, year in enumerate(unique_years):
        # # year_data = data.sel(time=data.time.dt.year == year)
        
        # # for lat_idx in range(n_lat):
            # # for lon_idx in range(n_lon):
                # # # 获取当前格点的时间序列
                # # if n_lat > 1 and n_lon > 1:
                    # # grid_data = year_data.isel(lat=lat_idx, lon=lon_idx).values
                # # elif n_lat > 1:
                    # # grid_data = year_data.isel(lat=lat_idx).values
                # # elif n_lon > 1:
                    # # grid_data = year_data.isel(lon=lon_idx).values
                # # else:
                    # # # 标量情况，直接获取值
                    # # grid_data = year_data.values
                
                # # # 确保数据是一维数组（时间序列）
                # # if grid_data.ndim > 1:
                    # # # 尝试将多维数组压缩为一维
                    # # grid_data = grid_data.flatten()
                
                # # # 检测干旱事件
                # # events = []
                # # current_event = []
                
                # # # 打印数据以帮助调试
                # # print(f"Year {year}, Lat {lat_idx}, Lon {lon_idx}: Data shape={grid_data.shape}, Min={np.min(grid_data)}, Max={np.max(grid_data)}")
                
                # # for i, value in enumerate(grid_data):
                    # # if value < threshold:
                        # # current_event.append((i, value))
                    # # else:
                        # # if len(current_event) >= duration_threshold:
                            # # events.append(current_event)
                        # # current_event = []
                
                # # # 检查最后一个事件
                # # if len(current_event) >= duration_threshold:
                    # # events.append(current_event)
                
                # # # 计算并存储统计数据
                # # if events:
                    # # durations = [len(event) for event in events]
                    # # severities = [sum(-val for _, val in event) for event in events]
                    # # annual_stats['max_duration'][year_idx, lat_idx, lon_idx] = max(durations)
                    # # annual_stats['max_severity'][year_idx, lat_idx, lon_idx] = max(severities)
    
    # # return annual_stats

# # # 数据加载和预处理
# # spi_data = xr.load_dataset('E:\chorm_download\spei03.nc')
# # CN_data = spi_data.sel(lat=slice(15,55), lon=slice(70,140))
# # start_date = '1960-01-01'
# # end_date = '2020-12-31'
# # CN_data = CN_data.sel(time=slice(start_date, end_date))
# # sample_data = CN_data['spei'].sel(lon=114.25, lat=30.75, method='nearest').expand_dims(['lat','lon'])
# # print(sample_data)

# # # 检测干旱事件
# # drought_events = detect_annual_events(
    # # data=sample_data,
    # # threshold=-1.0,
    # # duration_threshold=1
# # )

# # # 提取时间和SPEI值
# # times = sample_data.time.values
# # spei_values = sample_data.values.flatten()

# # # 创建DataFrame
# # df = pd.DataFrame({
    # # 'time': pd.to_datetime(times),
    # # 'spei': spei_values
# # })

# # # 按时间排序
# # df = df.sort_values('time')

# # # 保存为CSV文件
# # csv_path = r'spei_timeseries1.csv'
# # df.to_csv(csv_path, index=False)

# # # 打印结果
# # print("\n最终结果:")
# # for year in drought_events.year.values:
    # # duration = drought_events['max_duration'].sel(year=year).values
    # # severity = drought_events['max_severity'].sel(year=year).values
    # # print(f"{year}年: 最大持续时间={duration}, 最大严重程度={severity}")

