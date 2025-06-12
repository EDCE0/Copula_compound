import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import expon, gamma, kendalltau
from scipy.optimize import minimize
from scipy.integrate import quad
import os
import math
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# ==============================================================================
# 1. EVENT DETECTION FUNCTION (REFACTORED)
# ==============================================================================

def detect_annual_compound_events(tmp_series, spei_series, unique_years,
                                  tmp_threshold, spei_threshold, 
                                  duration_threshold):
    """
    Detects compound events for a single time series and returns annual maxima
    and all individual event characteristics.
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
            grid_annual_max_severity[year_idx] = max(severities)

    return grid_annual_max_duration, grid_annual_max_severity, np.array(all_durations), np.array(all_severities)

# ==============================================================================
# 2. PARALLEL PROCESSING FUNCTIONS
# ==============================================================================

def process_single_grid_point(args):
    """
    处理单个网格点的函数，用于并行计算
    """
    lat_idx, lon_idx, tmp_data_chunk, spei_data_chunk, unique_years, tmp_threshold, spei_threshold, duration_threshold = args
    
    try:
        # Select the time series for the current grid point
        tmp_grid_series = tmp_data_chunk.isel(lat=0, lon=0)  # 已经是单个网格点的数据
        spei_grid_series = spei_data_chunk.isel(lat=0, lon=0)
        
        # Call the detection function
        grid_ann_max_d, grid_ann_max_s, grid_all_d, grid_all_s = detect_annual_compound_events(
            tmp_series=tmp_grid_series,
            spei_series=spei_grid_series,
            unique_years=unique_years,
            tmp_threshold=tmp_threshold,
            spei_threshold=spei_threshold,
            duration_threshold=duration_threshold
        )
        
        # Calculate lambda for this grid point
        num_events_in_grid = len(grid_all_d)
        num_years_in_grid = len(unique_years)
        local_lambda = num_events_in_grid / num_years_in_grid if num_years_in_grid > 0 else 0
        
        return {
            'lat_idx': lat_idx,
            'lon_idx': lon_idx,
            'ann_max_duration': grid_ann_max_d,
            'ann_max_severity': grid_ann_max_s,
            'all_durations': grid_all_d,
            'all_severities': grid_all_s,
            'lambda': local_lambda
        }
    except Exception as e:
        print(f"Error processing grid point ({lat_idx}, {lon_idx}): {e}")
        return {
            'lat_idx': lat_idx,
            'lon_idx': lon_idx,
            'ann_max_duration': np.full(len(unique_years), np.nan),
            'ann_max_severity': np.full(len(unique_years), np.nan),
            'all_durations': np.array([]),
            'all_severities': np.array([]),
            'lambda': 0
        }

def prepare_grid_point_data(tmp_data, spei_data, unique_years, tmp_threshold, spei_threshold, duration_threshold):
    """
    准备并行处理的数据
    """
    tasks = []
    for lat_idx in range(len(tmp_data.lat)):
        for lon_idx in range(len(tmp_data.lon)):
            # 提取单个网格点的数据
            tmp_chunk = tmp_data.isel(lat=lat_idx, lon=lon_idx).expand_dims({'lat': [tmp_data.lat.values[lat_idx]]})
            spei_chunk = spei_data.isel(lat=lat_idx, lon=lon_idx).expand_dims({'lat': [spei_data.lat.values[lat_idx]]})
            
            tasks.append((lat_idx, lon_idx, tmp_chunk, spei_chunk, unique_years, 
                         tmp_threshold, spei_threshold, duration_threshold))
    
    return tasks

# ==============================================================================
# 3. COPULA AND RETURN PERIOD FUNCTIONS
# ==============================================================================

def fit_marginal_distributions(data, dist_type):
    if len(data) < 10:
        return None
    try:
        if dist_type == 'expon':
            loc, scale = expon.fit(data, floc=0)
            return {'loc': loc, 'scale': scale}
        elif dist_type == 'gamma':
            a, loc, scale = gamma.fit(data, floc=0)
            return {'a': a, 'loc': loc, 'scale': scale}
        else:
            raise ValueError("Unsupported distribution type.")
    except Exception:
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
    if tau >= 1.0: return np.inf
    theta = 1 / (1 - tau)
    return max(theta, 1.0)

def calculate_joint_return_period(duration_val, severity_val,
                                  marginal_params_d, marginal_params_s,
                                  dist_type_d, dist_type_s, copula_theta,
                                  lambda_val):
    """Calculates the joint return period in years."""
    if np.isnan(duration_val) or np.isnan(severity_val) or lambda_val == 0:
        return np.nan
        
    u_transform = get_cdf_value(duration_val, marginal_params_d, dist_type_d)
    v_transform = get_cdf_value(severity_val, marginal_params_s, dist_type_s)

    if np.isnan(u_transform) or np.isnan(v_transform):
        return np.nan

    joint_exceedance_prob = gumbel_copula_cdf(1 - u_transform, 1 - v_transform, copula_theta)

    if joint_exceedance_prob == 0:
        return np.inf
    
    return 1 / (lambda_val * joint_exceedance_prob)

# ==============================================================================
# 4. PARALLEL RETURN PERIOD CALCULATION
# ==============================================================================

def calculate_return_periods_for_grid(args):
    """
    计算单个网格点的重现周期
    """
    lat_idx, lon_idx, ann_max_duration, ann_max_severity, lambda_val, \
    global_marginal_params_duration, global_marginal_params_severity, global_copula_theta = args
    
    return_periods = np.full(len(ann_max_duration), np.nan)
    
    if lambda_val == 0:
        return lat_idx, lon_idx, return_periods
    
    for year_idx in range(len(ann_max_duration)):
        duration_val = ann_max_duration[year_idx]
        severity_val = ann_max_severity[year_idx]
        
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
            lambda_val=lambda_val
        )
        return_periods[year_idx] = rp
    
    return lat_idx, lon_idx, return_periods

# ==============================================================================
# 5. MAIN EXECUTION SCRIPT WITH PARALLEL PROCESSING
# ==============================================================================

def main():
    start_time = time.time()
    
    # --- Data Loading and Preprocessing ---
    DATA_PATH = 'I:/temp/CN_max_tmp_1961_2022.nc' 
    SPEI_DATA_PATH = 'I:/temp/SPEI_1961_2022.nc' 
    
    try:
        full_dataset = xr.load_dataset(DATA_PATH)
        SPEI_full_dataset = xr.load_dataset(SPEI_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found")
        return

    # Rename coordinates for consistency if they exist
    if 'longitude' in full_dataset.coords:
        full_dataset = full_dataset.rename({'longitude': 'lon'})
    if 'latitude' in full_dataset.coords:
        full_dataset = full_dataset.rename({'latitude': 'lat'})
        
    tmp_data = full_dataset['t2m'].sel(lat=slice(15, 35), lon=slice(70, 105))
    spei_data = SPEI_full_dataset['spei'].sel(lat=slice(15, 35), lon=slice(70, 105))

    print(f"Processing data with shape: {tmp_data.shape}")
    unique_years = np.unique(tmp_data.time.dt.year.values)

    # --- Initialize storage arrays ---
    coords = {'year': unique_years, 'lat': tmp_data.lat.values, 'lon': tmp_data.lon.values}
    all_years_duration = xr.DataArray(np.nan, dims=['year', 'lat', 'lon'], coords=coords)
    all_years_severity = xr.DataArray(np.nan, dims=['year', 'lat', 'lon'], coords=coords)
    all_years_return_period = xr.DataArray(np.nan, dims=['year', 'lat', 'lon'], coords=coords)
    coords_spatial = {'lat': tmp_data.lat.values, 'lon': tmp_data.lon.values}
    all_locations_lambda = xr.DataArray(np.nan, dims=['lat', 'lon'], coords=coords_spatial)

    # --- Parallel processing for event detection ---
    print("\nPreparing parallel processing tasks...")
    tasks = prepare_grid_point_data(tmp_data, spei_data, unique_years, 
                                   305.15, -1.0, 3)
    
    print(f"Processing {len(tasks)} grid points using {cpu_count()} CPU cores...")
    
    # 使用进程池进行并行计算
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_single_grid_point, tasks)
    
    print("Parallel event detection completed. Collecting results...")
    
    # --- Collect results ---
    global_all_event_durations = []
    global_all_event_severities = []
    
    for result in results:
        lat_idx = result['lat_idx']
        lon_idx = result['lon_idx']
        
        # Store annual maxima
        all_years_duration[:, lat_idx, lon_idx] = result['ann_max_duration']
        all_years_severity[:, lat_idx, lon_idx] = result['ann_max_severity']
        all_locations_lambda[lat_idx, lon_idx] = result['lambda']
        
        # Collect all events for global fitting
        if len(result['all_durations']) > 0:
            global_all_event_durations.extend(result['all_durations'])
            global_all_event_severities.extend(result['all_severities'])

    # --- Global fitting ---
    print("\nFitting global statistical models...")
    global_all_event_durations = np.array(global_all_event_durations)
    global_all_event_severities = np.array(global_all_event_severities)

    global_marginal_params_duration = fit_marginal_distributions(global_all_event_durations, 'expon')
    global_marginal_params_severity = fit_marginal_distributions(global_all_event_severities, 'gamma')

    if global_marginal_params_duration is None or global_marginal_params_severity is None:
        print("Could not fit global marginal distributions. Not enough data. Exiting.")
        return

    print(f"Global Duration marginal (Exponential) params: {global_marginal_params_duration}")
    print(f"Global Severity marginal (Gamma) params: {global_marginal_params_severity}")

    global_u_data = get_cdf_value(global_all_event_durations, global_marginal_params_duration, 'expon')
    global_v_data = get_cdf_value(global_all_event_severities, global_marginal_params_severity, 'gamma')

    valid_indices = ~np.isnan(global_u_data) & ~np.isnan(global_v_data)
    global_copula_theta = fit_gumbel_copula(global_u_data[valid_indices], global_v_data[valid_indices])
    print(f"Fitted Global Gumbel Copula theta: {global_copula_theta:.4f}")

    # --- Parallel calculation of return periods ---
    print("\nCalculating return periods using parallel processing...")
    
    rp_tasks = []
    for lat_idx in range(len(tmp_data.lat)):
        for lon_idx in range(len(tmp_data.lon)):
            lambda_val = all_locations_lambda.values[lat_idx, lon_idx]
            ann_max_duration = all_years_duration.values[:, lat_idx, lon_idx]
            ann_max_severity = all_years_severity.values[:, lat_idx, lon_idx]
            
            rp_tasks.append((lat_idx, lon_idx, ann_max_duration, ann_max_severity, lambda_val,
                           global_marginal_params_duration, global_marginal_params_severity, 
                           global_copula_theta))
    
    with Pool(processes=cpu_count()) as pool:
        rp_results = pool.map(calculate_return_periods_for_grid, rp_tasks)
    
    # Store return period results
    for lat_idx, lon_idx, return_periods in rp_results:
        all_years_return_period[:, lat_idx, lon_idx] = return_periods

    # --- Save results ---
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

    end_time = time.time()
    print(f"\nProcessing complete. Total time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
