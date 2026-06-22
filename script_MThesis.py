# %%
import pandas as pd

import numpy as np

from scipy.signal import savgol_filter, welch, butter, filtfilt
from scipy.stats import pearsonr, norm, laplace
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut

# %%
# 1. LOAD GAS BALANCE DATA
# ---------------------------------------------------------
# From CSV
df = pd.read_csv("path\\to\\file\\gas_balance.csv", 
                 sep=';', decimal=',', parse_dates=["date"], dayfirst=True)

# (!) The first point corresponds to inocculation the 02.05.2025 at 11:21:10
# (!) The change from ramp-up to production is on the 07.05.2025 at 16:11
# (!) The last point is at 300h afterwards because of off-gas filter blocked

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Extract date and create a masterclock for all databases
df["time"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M") 

t0 = df["time"].min()   # Masterclock --> beginning of bioprocess

t_prod0 = df.loc[df["date"] == pd.Timestamp("2025-05-07 16:11"), "date"].iloc[0] # Date and time of beginning of production phase
t_prod0_h = (t_prod0 - t0).total_seconds() / 3600       # to h

df["t_[h]"] = df["t_[s]"] / 3600   # To also have time in h
df = df.sort_values("t_[h]")    #Make sure the time is ascending (sanity check)

# Map and extract columns: O2 information
t_s  = df["t_[s]"].to_numpy() /3600      # h
DO_perc = df["pO2_[perc-sat]"].to_numpy() /100    # %-sat to fraction
y_out  = df["EX-O2_[perc-vol]"].to_numpy() /100      # %-vol to fraction
SPG_o2   = df["SPG-O2_[slpm]"].to_numpy()
SPG_co2  = df["SPG-CO2_[slpm]"].to_numpy()  # Necessary for the global mass balance (total Q)

# %% 
# 2. LOAD VCD
# ---------------------------------------------------------
vcd_df = pd.read_csv("path\\to\\file\\vcd.csv", 
                     sep=';', decimal=',', parse_dates=["date"], dayfirst=True)
# Data from: Achieving target DO after inocculation: 02.05.2025 11:48:00

# Ensure column names match your file
vcd_df.columns = vcd_df.columns.str.strip()

# Extract date
vcd_df["time"] = pd.to_datetime(vcd_df["date"], format="%Y-%m-%d %H:%M") 
vcd_df["t_[h]"] = (vcd_df["time"] - t0).dt.total_seconds() / 3600   # Create t_[h] --> same as: vcd_df["t_[min]"] / 60 in this case (mins)
vcd_df = vcd_df.sort_values("t_[h]")    # Sort time ascending

# Replace negative VCD values with zero
vcd_df["VCD_[E6/ml]"] = vcd_df["VCD_[E6/ml]"].clip(lower=0)

# Extract columns
t_vcd = vcd_df["t_[min]"].to_numpy() / 60   # VCD timestamps [h]
VCD = vcd_df["VCD_[E6/ml]"].to_numpy()      # VCD measurements

# For latter analysis
VCD_min = pd.Series(VCD * 1e6 * 1000, index=pd.to_timedelta(np.arange(len(VCD)), unit='m'))
VCD_safe = np.maximum(VCD_min, 0.1 * VCD_min.iloc[0])

# %% 
# 2.2. LOAD OFFLINE VCD TIMEPOINTS
# ---------------------------------------------------------
vcd_offline = pd.read_csv("path\\to\\file\\vcd_offline.csv", sep=';')

# Ensure column names match your file
vcd_offline.columns = vcd_offline.columns.str.strip()

# Replace commas with dots and convert all columns to numeric
for col in vcd_offline.columns:
    # Remove any extra spaces
    vcd_offline[col] = vcd_offline[col].astype(str).str.replace(",", ".", regex=False).str.strip()
    # Convert to numeric (non-convertible values become NaN)
    vcd_offline[col] = pd.to_numeric(vcd_offline[col], errors='coerce')

# Extract columns
t_s2 = vcd_offline["t_[s]"]/3600         # h
vcd_reps = vcd_offline[
    ["VCD1_[E6/ml]", "VCD2_[E6/ml]", "VCD3_[E6/ml]"]
].to_numpy()

# Define sigmoidal
def sigmoid(x, bottom, top, x0, k):
    return bottom + (top - bottom) / (1 + np.exp(-(x - x0) / k))

# Flatten data for fitting
t_fit = np.repeat(t_s2.values, vcd_reps.shape[1])
vcd_fit = vcd_reps.flatten()

# Define piecewise exponential-linear model (CONTINUOUS VERSION)
def piecewise_exp_lin(x, A, k, t0, m, b):
    # exponential phase
    exp_part = A * np.exp(k * (x - t0)) + b

    # value at switching point ensures continuity
    y_t0 = A + b

    # linear phase anchored at t0 (continuous transition)
    lin_part = y_t0 + m * (x - t0)

    return np.where(x <= t0, exp_part, lin_part)

# Initial parameter guesses (sigmoidal)
p0 = [
    np.nanmin(vcd_fit),        # bottom
    np.nanmax(vcd_fit),        # top
    np.nanmedian(t_s2),       # x0 (inflection time)
    (t_s2.max() - t_s2.min()) / 10   # slope
]

params, cov = curve_fit(sigmoid, t_fit, vcd_fit, p0=p0)

t_smooth = np.linspace(t_s2.min(), t_s2.max(), 400)
vcd_smooth = sigmoid(t_smooth, *params)

plt.figure(figsize=(10,5))

# Initial guesses (piecewise model)
p0_pw = [
    np.nanmax(vcd_fit) - np.nanmin(vcd_fit),   # A
    0.05,                                      # k (growth rate)
    np.nanmedian(t_s2),                        # t0
    0.1,                                       # m (linear slope)
    np.nanmin(vcd_fit)                         # b (offset)
]

params_pw, cov_pw = curve_fit(piecewise_exp_lin, t_fit, vcd_fit, p0=p0_pw)

vcd_pw_smooth = piecewise_exp_lin(t_smooth, *params_pw)

# Plot replicates
plt.figure(figsize=(10,5))
for i in range(vcd_reps.shape[1]):
    plt.scatter(t_s2, vcd_reps[:, i], alpha=0.8, label=f'Offline VCD{i+1}')

# Plot fit
plt.plot(t_smooth, vcd_smooth, label='Offline Sigmoid fit', color="orange")
plt.plot(t_smooth, vcd_pw_smooth, label='Piecewise exp-linear fit', color="green")

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.scatter(vcd_df["t_[min]"]/60, vcd_df["VCD_[E6/ml]"], s=1, alpha=0.3, label="Online VCD", color="tab:blue")

plt.xlabel('Time [h]')
plt.ylabel('VCD [E6/ml]')
plt.ylim(0,35)
plt.legend()
plt.show()

# Extract parameters (sigmoid)
bottom, top, t_in, k = params

print(f"Initial VCD : {bottom:.2f} E6/ml")
print(f"Max VCD     : {top:.2f} E6/ml")
print(f"Inflection  : {t_in:.2f} h")
print(f"Slope       : {k:.2f} h")

# Extract parameters (piecewise)
A_pw, k_pw, t0_pw, m_pw, b_pw = params_pw

print(f"\nPiecewise model:")
print(f"A (exp amplitude): {A_pw:.2f}")
print(f"k (exp rate)     : {k_pw:.4f}")
print(f"t0 (switch time) : {t0_pw:.2f} h")
print(f"m (linear slope) : {m_pw:.4f}")
print(f"b (offset)       : {b_pw:.2f}")

# SG filtering
from scipy.signal import savgol_filter
window_minutes = 1441   # must be odd → ~24 hour window
poly_order = 2

vcd_sg = savgol_filter(vcd_df["VCD_[E6/ml]"].values, window_length=window_minutes, polyorder=poly_order)

# Plot
plt.figure(figsize=(10,5))
plt.scatter(vcd_df["t_[min]"]/60, vcd_df["VCD_[E6/ml]"], s=1, alpha=0.3, label="Online VCD", color="tab:blue")
plt.plot(vcd_df["t_[min]"]/60, vcd_sg, color="orange", linewidth=2, label="SG filtered (≈24 h window)")
plt.xlabel("Time [h]")
plt.ylabel("VCD [E6/mL]")
plt.legend()
plt.show()

# %%
# 3. CONSTANTS
# ---------------------------------------------------------
# Bioreactor volume
V_t = 100       # L
V_l = 0.8*V_t   # L 

# Overlay: constant
OVL_air = 8     # slpm
OVL_co2 = 0     # slpm
OVL_o2 = 0      # slpm

# Sparging
SPG_air = 2.015 #slpm

y_O2_air = 0.2095   # O2 fraction in air
p_br = 1.09869233   # atm
Mw_o2 = 32          # g/mol
T = 295             # K (22°C)
R = 0.082057        # atm·L/(K·mol)

# Ideal gas equation: pV = nRT --> Molar volume of gas Vm = V/n
Vm = R*T/p_br                  # L/mol (volume that takes 1 mol O2)
C_O2_gas = p_br*Mw_o2/(R*T)    # mg/L (mg O2 per L gas)

dt = np.median(np.diff(t_s))  # hours

seed = 5000 
rng = np.random.default_rng(seed)

# 4. UNCERTAINTIES
# ---------------------------------------------------------
# Statistical uncertainty (sigma A):
# In type A uncertainty, no division by sqrt(3) is neccessary --> std() already accounts for it

window_sigma = 360          # corresponds to 1h

sigmaA_yO2out = pd.Series(y_out).rolling(int(window_sigma), min_periods=1, center=True).std().bfill().to_numpy()       

sigmaA_OVL_air = 0.01               # It is set cte in this script; set a minimal amount
sigmaA_SPG_air = 0.01               # It is set cte in this script; set a minimal amount
sigmaA_OVL_co2 = 1e-6               # It is set cte in this script; set a minimal amount
sigmaA_SPG_co2 = pd.Series(SPG_co2).rolling(int(window_sigma), min_periods=1, center=True).std().bfill().to_numpy()
sigmaA_OVL_o2 = 1e-6                # It is set cte in this script; set a minimal amount
sigmaA_SPG_o2 = pd.Series(SPG_o2).rolling(int(window_sigma), min_periods=1, center=True).std().bfill().to_numpy()

sigmaA_VCD = pd.Series(VCD).rolling(int(window_sigma/60), min_periods=1, center=True).std().bfill().to_numpy()

sigmaA_Vl = 1e-6
#sigmaA_feed_weight is below in Section 9

sigmaA_gluc = 1e-3
sigmaA_lac = 1e-3

# Equiment uncertainty (sigma B):
    # Rectangular: sigma = a/sqrt(3)
    # a = tolerance band (maxium allowed error)

a_abs_yO2out = (0.002*0.5)    # Absolute error: <0.2% FullScale, BioPAT Xgas, Gaussian (sensor)
a_rel_yO2out = (0.03*y_out)    # Relative error: 3% reading, BioPAT Xgas, Gaussian (sensor)
sigmaB_yO2out = np.sqrt((a_abs_yO2out/np.sqrt(3))**2 + (a_rel_yO2out/np.sqrt(3))**2)    # Error combination

sigmaB_OVL_air = (0.01*20)/np.sqrt(3)        # +- 1% FS
sigmaB_SPG_air = (0.01*20)/np.sqrt(3)        # +- 1% FS
sigmaB_OVL_co2 = (0.01*3)/np.sqrt(3)         # +- 1% FS
sigmaB_SPG_co2 = (0.01*3)/np.sqrt(3)         # +- 1% FS
sigmaB_OVL_o2 = (0.01*20)/np.sqrt(3)         # +- 1% FS
sigmaB_SPG_o2 = (0.01*20)/np.sqrt(3)         # +- 1% FS

sigmaB_VCD = 0.03*VCD/np.sqrt(3)             # [E6/ml], ABER-Futura System: accuracy (tolerance, a): < 3% reading. Gaussian (sensor)
sigmaB_Vl = (0.01*150)/np.sqrt(3)            # +- 1% FS
sigmaB_feed_weight = 0.5/np.sqrt(3)          # [kg], From calibration sheet --> sensor

#sigmaB_gluc, sigmaB_lac are given below in Section 11

# Combined uncertainty (sigma = sqrt(sigmaA**2+sigmaB**2)):
sigma_yO2out = np.sqrt(sigmaA_yO2out**2+sigmaB_yO2out**2)

sigma_OVL_air = np.sqrt(sigmaA_OVL_air**2+sigmaB_OVL_air**2)
sigma_SPG_air = np.sqrt(sigmaA_SPG_air**2+sigmaB_SPG_air**2)
sigma_OVL_co2 = np.sqrt(sigmaA_OVL_co2**2+sigmaB_OVL_co2**2)
sigma_SPG_co2 = np.sqrt(sigmaA_SPG_co2**2+sigmaB_SPG_co2**2)
sigma_OVL_o2 = np.sqrt(sigmaA_OVL_o2**2+sigmaB_OVL_o2**2)
sigma_SPG_o2 = np.sqrt(sigmaA_SPG_o2**2+sigmaB_SPG_o2**2)

sigma_Vl = np.sqrt(sigmaA_Vl**2+sigmaB_Vl*2)
sigma_VCD = np.sqrt(sigmaA_VCD**2+sigmaB_VCD**2)
sigma_VCD_min = pd.Series(sigma_VCD, index=pd.to_timedelta(np.arange(len(VCD)), unit='m'))

# %%
# 5.1 PROCESS COLUMNS AND PLOT RAW DATA
# ---------------------------------------------------------
# Replace all negative flow readings with 0 (because real flow cannot be negative)
flows = [SPG_co2, SPG_o2]
flows = [np.clip(f, 0, None) for f in flows]
SPG_co2_v, SPG_o2_v = flows

# Compute total gas flow and its uncertainty: Q_gas = sum of all OVL + all SPG 
BASE_FLOW = OVL_air + OVL_co2 + OVL_o2 + SPG_air    # Constant part of the flow
Qg_slpm = BASE_FLOW + SPG_co2_v + SPG_o2_v
Qg = np.where(Qg_slpm == 0, np.nan, Qg_slpm)   # slpm, safe to avoid dividing by 0 later
df["Qg_[slpm]"] = Qg

sigma_Qg = np.sqrt((sigma_OVL_air**2 + sigma_OVL_co2**2 + sigma_OVL_o2**2 + sigma_SPG_air**2 +
           sigma_SPG_co2**2 + sigma_SPG_o2**2))    # Linear noise propagation
df["sigma_Qg"] = sigma_Qg

# Plot DO
plt.figure(figsize=(8,5))
plt.plot(t_s, DO_perc*100, label = "Raw DO", alpha = 0.9)
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
#plt.xlim(250, 251)
#plt.ylim(40, 60)
plt.ylabel("DO [perc-sat]")
plt.axvspan(104, 118, color='grey', alpha=0.2, lw=0)
plt.axvspan(122, 142, color='grey', alpha=0.2, lw=0)
plt.axvspan(210, 240, color='grey', alpha=0.2, lw=0)
plt.legend()
plt.show()

# Plot y_out +- sigma
plt.figure(figsize=(8,5))
plt.plot(t_s, y_out*100, label = "O₂ out", alpha = 0.8)
plt.fill_between(t_s, y_out*100 + sigma_yO2out*100, y_out*100 - sigma_yO2out*100, label = "σ", alpha = 0.3, color="#24B8D6")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
plt.ylabel("yO₂ [%-vol]")
plt.legend()
plt.title("")
plt.show()

# Plot SPG-o2 +- sigma
plt.figure(figsize=(8,5))
plt.plot(t_s, SPG_o2, label = "SPG O₂", alpha = 0.8)
plt.fill_between(t_s, SPG_o2 + sigma_SPG_o2, SPG_o2 - sigma_SPG_o2, label = "σ", alpha = 0.3, color="#24B8D6")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
plt.ylabel("Flow rate (Q) [slpm]")
plt.legend()
plt.title("")
plt.show()

# Plot VCD +- sigma
plt.figure(figsize=(8,5))
plt.plot(t_vcd, VCD, label = "VCD", alpha = 0.8)
plt.fill_between(t_vcd, VCD + sigma_VCD, VCD - sigma_VCD, label = "σ", alpha = 0.3, color="#24B8D6")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
plt.ylabel("VCD [E6 vc/ml]")
plt.legend()
plt.title("")
plt.show()

#%%
# 5.2 RAW OUR CALCULATION
# ---------------------------------------------------------
y_in_raw = (y_O2_air * (OVL_air + SPG_air) + OVL_o2 + SPG_o2_v) / Qg
# Compute dy 
df["y_out"] = y_out
df["sigma_yout"] = sigma_yO2out
dy = y_in_raw - y_out
df["dy"] = dy

# Calculate raw OUR to compare
OUR_raw= (dy*Qg/Vm*60*1000)/V_l
df["OUR_raw_[mmol/h/L]"] = OUR_raw #mmol/h L

#%%
# 5.3 PERICELLULAR CONCENTRATION
# ---------------------------------------------------------
# Calculate qO2 raw
OUR_raw_min = (             # [mmol/h/L]
    df.assign(t_s=pd.to_timedelta(df["t_[s]"], unit="s"))
      .set_index("t_s")["OUR_raw_[mmol/h/L]"]       #VERY IMPORTANT: choose the OUR to use FROM df (raw, smooth_SG, smooth_KF, etc.)
      .resample("1min")
      .mean()
)

VCD_min = pd.Series(VCD * 1e6 * 1000, index=pd.to_timedelta(np.arange(len(VCD)), unit='m'))

qO2_raw = (OUR_raw_min* 1e9 * 24) / VCD_min   # [pmol/vc/d]

# Offline diameter measurement
d_max = 19.6    # um; offline measured
d_min = 15.6    # um; offline measured

def pericellular_conc(d):
    r = (d/2)*1e-6                           # m
    D_O2 = 2e-9                              # m2/s; diffusion coefficient of O2 in water at 37°C (approx)

    qO2_peri = qO2_raw*1e-12 / 86400             # mol/vc/s

    dcO2 = qO2_peri/(4*np.pi*D_O2*r)*1000   # uM
    cO2_peri = np.minimum(100 - dcO2, 100)                   # uM; 100 uM corresponds to 50% DO
                
    return cO2_peri

dO2_maxR = pericellular_conc(d_max)
dO2_minR = pericellular_conc(d_min)

plt.figure(figsize=(8,5))
plt.plot(t_vcd, dO2_maxR, label = "Max diameter (19.6 µm)", alpha = 0.5)
plt.plot(t_vcd, dO2_minR, label ="Min diameter (15.6 µm)", alpha = 0.5)
plt.ylabel("O₂ concentration [µM]")
plt.xlabel("Time [h]")
plt.legend()
plt.axhline(y=100, color = "#817E7C", linestyle= ":")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')

# %%
# 6.1 LOAD FEED, BLEED AND HARVEST
# -------------------------------------------
weight = pd.read_csv("path\\to\\file\\feed_weight.csv", 
                     sep=';', decimal=',', parse_dates=["date"], dayfirst=True)
#Starting from 02.05.2025 at 11:21:10 corresponding to 38948s for raw export of data (this is set to t=0s)
#Measured each 2s
# Just process feed because D = Feed = Bleed + Harvest

# Strip whitespace from column names
weight.columns = weight.columns.str.strip()

# Extract date
weight["time"] = pd.to_datetime(weight["date"], format="%Y-%m-%d %H:%M:%S")

# Create an average value per minute (measurements each 2s in such big volumes give mathematical problems)
weight = (
    weight.groupby("time", as_index=False)
          .mean()  # or use another aggregation
)

weight["t_[h]"] = (weight["time"] - t0).dt.total_seconds() / 3600         # Create t_[h]
weight = weight.sort_values("t_[h]")            # Sort time ascending

# Configuration parameters
min_bag_jump = 150     # kg (min weight change interpreted as bag change)

# Load data
feed_raw = weight["feed_[kg]"].to_numpy()   # kg
time_h = weight["t_[h]"]

if not weight["time"].is_monotonic_increasing:
    raise ValueError("Time must be strictly increasing!")

# Calculation of sigma B feed
feed_trend = pd.Series(feed_raw).rolling(30, center=True).mean()
sigmaA_feed_weight = np.nanstd(feed_raw - feed_trend)     # Signal - EMA as tendency 
sigma_feed = np.sqrt(sigmaA_feed_weight**2 + sigmaB_feed_weight**2)     # Combined uncertainty

# Functions
#-- Detect bag swaps
def detect_feed_swaps(weight):

    dw = np.diff(weight)
    
    swaps = np.where(dw > min_bag_jump)[0] + 1
    
    print("\n===== BAG SWAP DETECTION =====")
    print("Detected swaps at indices:", swaps)
    print("Total swaps:", len(swaps))
    
    return swaps

#-- Remove swap artifacts --> subsitute by NaNs
def mask_swaps(signal, swaps, window=5):
    """
    Replace values around bag swaps with NaN to reduce peaks.
    """
    
    clean = signal.astype(float).copy()
    
    for s in swaps:
        start = max(0, s-window)
        end = min(len(signal), s+window)
        clean[start:end] = np.nan
        
    print(f"Masked ±{window} samples around swaps")
    return clean

# Feed preprocess
feed_swaps = detect_feed_swaps(feed_raw)        # kg
feed_clean = mask_swaps(feed_raw, feed_swaps, window=15)    # Window = N samples around the bag swap; kg

# Monte Carlo propagation
N_mc = 5000  # number of Monte Carlo runs
dt_w = np.median(np.diff(time_h))   # hours
alpha_ema = 0.02  # EMA smoothing factor --> dimesionless --> N_eff = 2/alpha - 1 = 100

flows_mc = np.empty((N_mc, len(feed_clean)))

for i in range(N_mc):
    # Add noise to feed
    feed_noisy = feed_clean + rng.normal(0, sigma_feed, size=len(feed_clean))     # Addition of random Gaussian noise according to sigma_feed
    
    # Optional: mask swaps again to keep zeros
    invalid = np.isnan(feed_noisy[:-1]) | np.isnan(feed_noisy[1:])
    invalid = np.insert(invalid, 0, False)
    feed_noisy[invalid] = np.nan
    
    # Compute flow (derivative)
    flow_i = -np.gradient(feed_noisy, dt_w)         # Negative dfeed (consumption); kg/h ~ L/h
    
    # Set flow to 0 during NaNs (bag swaps)
    flow_i[invalid] = 0.0
    
    # EMA smoothing
    flow_i_smooth = pd.Series(flow_i).ewm(alpha=alpha_ema, adjust=False).mean().to_numpy()
    
    flows_mc[i] = flow_i_smooth

# Compute mean and std across MC runs
flow_smooth_mean = np.nanmean(flows_mc, axis=0)     # [kg/h ~ L/h]
sigma_flow_smooth  = np.nanstd(flows_mc, axis=0)

# Perfusion rate & uncertainty
D = np.clip(flow_smooth_mean / V_l, 0, None)        # [RV/h]
sigma_D = D * np.sqrt((sigma_flow_smooth/flow_smooth_mean)**2 + (sigma_Vl/V_l)**2)

CSPR = D / VCD_min      # [L/vc/h]
CSPR = CSPR * 1e9 * 24           # [nL/vc/d]
sigma_CSPR = CSPR * np.sqrt((sigma_D/np.maximum(D, 1e-12))**2 + (sigma_VCD_min/VCD_safe)**2) # No division by 0 --> min is 10% of first measurement

# Plot CSPR +- sigma
plt.figure(figsize=(8,5))
plt.fill_between(t_vcd, CSPR - sigma_CSPR, CSPR + sigma_CSPR, alpha=0.5, label="σ", color="#24B8D6")
plt.plot(t_vcd, CSPR, label="CSPR")
plt.ylabel("CSPR [nL/vc/d]")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.axvspan(104, 118, color='grey', alpha=0.2, lw=0)
plt.axvspan(122, 142, color='grey', alpha=0.2, lw=0)
plt.axvspan(210, 240, color='grey', alpha=0.2, lw=0)
plt.ylim(-0.1,0.5)
plt.xlabel("Time [h]")
plt.legend()
plt.show()

#%%
# 6.2 OFFLINE FEED TO CONFIRM
# -------------------------------------------
feed_offline = pd.read_csv("path\\to\\file\\feed_offline.csv", 
                     sep=';', decimal=',', parse_dates=["date"], dayfirst=True)

# Strip whitespace from column names
feed_offline.columns = feed_offline.columns.str.strip()

feed_offline["time"] = pd.to_datetime(feed_offline["date"], format="%Y-%m-%d %H:%M")

feed_offline["t_[h]"] = (feed_offline["time"] - t0).dt.total_seconds() / 3600         # Create t_[h]

feed_offline["feed_newbag_[kg]"][0] = feed_clean[0]     # Give value to the first new bag 

# Plot for steps feed processing
plt.figure(figsize=(8,10))
# Plot: dFeed
plt.subplot(4,1,1)
plt.plot(time_h, feed_clean, label="", linewidth=1)
#plt.plot(time_h, feed_raw, label="", linewidth=1)
plt.fill_between(t_vcd, feed_raw - sigma_feed, feed_raw + sigma_feed, alpha=0.5, label="σ", color="#24B8D6")
plt.scatter(feed_offline["t_[h]"], feed_offline["feed_[kg]"], label = "Feed offline", zorder = 10, color = "red")
plt.scatter(feed_offline["t_[h]"], feed_offline["feed_discon_[kg]"], marker='x', label = "Disconnect", zorder = 11, color = "green")
plt.scatter(feed_offline["t_[h]"], feed_offline["feed_newbag_[kg]"], marker='x', label = "New bag", zorder = 11, color = "orange")
#plt.xlabel("Time [h]")
plt.ylabel("Feed [L]")
plt.legend()
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')

# Plot: Flow
plt.subplot(4,1,2)
plt.plot(time_h, flow_smooth_mean*24, linewidth=1, label = "")
plt.fill_between(t_vcd, flow_smooth_mean*24 - sigma_flow_smooth*24, flow_smooth_mean*24 + sigma_flow_smooth*24, alpha=0.5, label="σ", color="#24B8D6")
plt.scatter(feed_offline["t_[h]"], feed_offline["flow_[L/d]"], label = "Flow offline", color = "red", zorder=10)
#plt.xlabel("Time [h]")
plt.ylabel("Flow [L/d]")
plt.ylim(-350, 1200)
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.legend()

# Plot: Perfusion Rate (D)
plt.subplot(4,1,3)
plt.plot(time_h, D*24, label="", linewidth=1)
plt.fill_between(t_vcd, D*24 - sigma_D*24, D*24 + sigma_D*24, alpha=0.5, label="σ", color="#24B8D6")
plt.scatter(feed_offline["t_[h]"], feed_offline["flow_[RV/d]"], label = "D offline", color = "red", zorder=10)
#plt.xlabel("Time [h]")
plt.ylim(-5, 15)
plt.ylabel("Perfusion Rate (D) [RV/d]")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.legend()

# Plot: CSPR
plt.subplot(4,1,4)
plt.plot(time_h, CSPR, label="", linewidth=1)
plt.fill_between(t_vcd, CSPR - sigma_CSPR, CSPR + sigma_CSPR, alpha=0.5, label="σ", color="#24B8D6")
plt.scatter(feed_offline["t_[h]"], feed_offline["CSPR_[nL/vc/d]"], label = "CSPR offline", color = "red", zorder=10)
plt.xlabel("Time [h]")
plt.xlabel("Time [h]")
plt.ylim(-0.7,0.7)
plt.ylabel("CSPR [nL/vc/d]")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.legend()
plt.show()

# %%
# 7. SAVITZKY-GOLAY AND POLYNOMIAL FIT
# ---------------------------------------------------------------------
# SG parameters
wind = 241           # must be odd
poly_sg = 2           # 2 or 3 is typical for biosignals

OUR_SG = savgol_filter(OUR_raw, window_length=wind, polyorder=poly_sg)

plt.figure(figsize=(8,5))
plt.plot(t_s, OUR_raw, label="Raw OUR", alpha = 0.5)
plt.plot(t_s, OUR_SG, label="SG (window = 241, degree = 2)", color = "red")
plt.axvline(124.82, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
plt.ylabel("OUR [mmol/h·L]")
plt.legend()
plt.show()

# Trend metrics for SG
fs = 1 / dt     # sampling frequency [1/hour]
fc = 0.0085     # cutoff frequency [Hz] --> adjusted with Power Spectral Density (PSD) visually (~30.6 1/h)
b, a = butter(N=2, Wn=fc/(fs/2), btype='low')

OUR_sg_trend = filtfilt(b, a, OUR_SG)

NRMSE_trend_SG = np.sqrt(
    np.mean((OUR_sg_trend - OUR_raw)**2)
) / np.mean(OUR_raw)

corr_trend_SG = np.corrcoef(OUR_sg_trend, OUR_raw)[0,1]

print(f"NRMSE SG:", round(NRMSE_trend_SG,4))
print(f"Pearson correlation SG:", round(corr_trend_SG,4))

# Smoothness
smoothness_SG = np.var(np.diff(OUR_SG))

print(f"Smoothness SG:", round(smoothness_SG,4))

# Polynomial regression
time = t_s.reshape(-1,1)

degrees = [2, 3, 4, 5, 6, 7] 
results = {}

for d in degrees:
    # Transform time into polynomial features
    poly = PolynomialFeatures(degree=d)
    t_poly = poly.fit_transform(time)
    
    # Fit model
    model = LinearRegression()
    model.fit(t_poly, OUR_SG) #OUR_SG
    
    # Predict
    OUR_pred = model.predict(t_poly)
    
    # Compute R²
    r2 = r2_score(OUR_SG, OUR_pred) #OUR_SG
    results[d] = r2

    plt.figure(figsize=(8,5))
    plt.scatter(t_s, OUR_SG, s=1, alpha=1, color = "orange", label=f"Degree = {d}, R2 = {np.round(r2,4)}") #OUR_SG
    plt.plot(t_s, OUR_pred, c="#753030") #label="Polynomial fit", 
    plt.plot(t_s, OUR_raw, color = "tab:blue", alpha = 0.5, zorder = 0)    #, label = "Raw OUR"
    plt.xlabel("Time [h]")
    plt.ylabel("OUR [mmol/h·L]")
    plt.axvline(124.82, color='#cd5c5c', linestyle='--') #, label='Ramp-up → Production',
    plt.legend()
    plt.show()

#%%
# 8.0 UKF EVALUATION
# -----------------------------
def evaluate_filter(
    OUR_filt,
    sigma_OUR,
    innovation,
    S_all,
    z_meas=None,
    z_pred=None,
    OUR_gas_series=None,
    dim_z=3
):
    metrics = {}

    N = len(OUR_filt)

    
    # 1. NIS computation
    nis = np.zeros(N)

    for k in range(N):
        nu = innovation[k]
        S = S_all[k]

        # Regularization (important!)
        eps = 1e-6 * np.trace(S) / dim_z
        S_reg = S + eps * np.eye(dim_z)

        nis[k] = nu.T @ np.linalg.solve(S_reg, nu)

    metrics["nis"] = nis

    
    # 2. Trend metrics
    fs = 1 / dt     # sampling frequency [1/hour]
    fc = 0.0085     # cutoff frequency [Hz] --> adjusted with Power Spectral Density (PSD) visually (~30.6 1/h)
    b, a = butter(N=2, Wn=fc/(fs/2), btype='low')
    OUR_filt_trend = filtfilt(b, a, OUR_filt)               # OUR filtered
    OUR_ref_trend = filtfilt(b, a, OUR_gas_series)          # OUR raw

    metrics["NRMSE_trend"] = np.sqrt(np.mean((OUR_filt_trend - OUR_ref_trend)**2)) / np.mean(OUR_ref_trend)
    metrics["corr_trend"] = np.corrcoef(OUR_filt_trend, OUR_ref_trend)[0,1]         # Shape agreement

    print("------------------------------")
    print("TREND STATS (Butterworth filter)")
    print("------------------------------")
    print("Trend: NRMSE (~ 0):",  np.round(metrics["NRMSE_trend"],4))
    print("Trend: Correlation (> 0.9):",  np.round(metrics["corr_trend"],4))

    
    # 3. Physical metrics (OUR)
    # Uncertainty consistency (normalized error)
    metrics["SNE_OUR"] = np.mean((OUR_filt - OUR_gas_series)**2 / (sigma_OUR**2 + 1e-8))

    print("------------------------------")
    print("PHYSICAL CONSISTENCY")
    print("------------------------------")
    print("SNE_OUR (~ 1):",  np.round(metrics["SNE_OUR"],4)) # Actual error / predicted covariance

    
    # 4. Secondary metrics
    # ANIS (size of error vs uncertainty)
    mean_nis = np.mean(nis)
    metrics["ANIS"] = mean_nis / dim_z

    # Innovation bias (mean error)
    metrics["innovation_bias"] = np.mean(innovation, axis=0)

    metrics["smoothness"] = np.var(np.diff(OUR_filt))

    print("------------------------------")
    print("SECONDARY STATISTICS")
    print("------------------------------")
    print("ANIS (~ 1):", np.round(metrics["ANIS"],4))   # Normalized NIS to the mesurement dimension
    print("Innovation bias (~ 0):", np.round(metrics["innovation_bias"],4))
    print("Smoothness (middle-low):",  np.round(metrics["smoothness"],4))  # Not too much because of oversmoothing

    return metrics

# %%
# 8.1 (v0) UKF direct in precalculated OUR (OUR calculation no integrated in UKF)
# ------------------------------------------------------
# Define state-space models
def fx(x, dt):
    OUR, dOUR = x
    return np.array([
        OUR + dOUR * dt,
        dOUR
    ])

def hx(x):
    OUR, _ = x
    return np.array([OUR])

# Sigma points
points = MerweScaledSigmaPoints(
    n=2, alpha=1e-3, beta=2.0, kappa=0.0
)

dim_z = 1

# Build UKF
ukf = UKF(
    dim_x=2,
    dim_z=dim_z,
    fx=fx,
    hx=hx,
    dt=dt,
    points=points
)

# Initial state
ukf.x = np.array([
    df["OUR_raw_[mmol/h/L]"].iloc[0],   # initial OUR
    0.0                                 # initial slope
])

# Initial covariance
ukf.P = np.diag([1.0,                   # OUR uncertainty
                 0.01])                 # slope uncertainty

# Measurement noise estimation
# Use first-difference to estimate sensor + algebra noise
OUR_diff = df["OUR_raw_[mmol/h/L]"].diff().dropna()
R0 = max(np.var(OUR_diff), 1e-3)    # Min fix

# Adaptive noise scaling factor
alpha_R = 0.02 

# Process noise (biological smoothness)
ukf.Q = np.diag([
    1e-4,    # OUR process noise
    1e-6     # slope process noise
])

# Run filter
OUR_UKF = np.zeros(len(df))
OUR_dot = np.zeros(len(df))
sigma_OUR = np.zeros(len(t_s))
innovation_all = np.zeros((len(t_s), dim_z))
S_all = np.zeros((len(t_s), dim_z, dim_z))
z_pred_all = np.zeros((len(t_s), dim_z))
z_meas_all = np.zeros((len(t_s), dim_z))

for k, z in enumerate(df["OUR_raw_[mmol/h/L]"].to_numpy()):

    # Predict
    ukf.predict()

    # Adaptive measurement noise:
    # controller oscillations grow with OUR magnitude
    ukf.R = np.array([[R0 + alpha_R * ukf.x[0]**2]])

    # Update (if measurement is valid)
    if np.isfinite(z):
        ukf.update([z])

    # Store metrics
    OUR_UKF[k] = ukf.x[0]
    OUR_dot[k] = ukf.x[1]
    sigma_OUR[k] = np.sqrt(ukf.P[0,0])
    innovation_all[k,:] = ukf.y
    S_all[k,:,:] = ukf.S
    z_pred_all[k,:] = hx(ukf.x)
    z_meas_all[k,:] = z

df["OUR_v0_[mmol/h/L]"] = np.asarray(OUR_UKF, dtype=float)

# Metrics UKF
metrics_UKF_v0 = evaluate_filter(
    OUR_filt=df["OUR_v0_[mmol/h/L]"],
    sigma_OUR=sigma_OUR[k],
    innovation=innovation_all,
    S_all=S_all,
    z_meas=z_meas_all,
    z_pred=z_pred_all,
    OUR_gas_series=OUR_raw,
    dim_z = dim_z
)

# %%  
# 8.2 (v1) UKF FOR RAW INPUTS without delay and soft dynamics
# ---------------------------------------------------------
# Conversion constant for OUR
K_OUR = 60 * 1000 / (V_l * Vm)   

# Time step
dt = np.median(np.diff(t_s))  # hours

#Process model: Prediction about how the state evolves over time
def fx(x, dt):
    # State vector
    yin, yout, SPG_O2, OUR, dyin, dyout, dSPG = x

    # Smooth dynamics
    yin_n  = yin  + dyin  * dt
    yout_n = yout + dyout * dt
    SPG_O2_n = max(SPG_O2 + dSPG * dt, 0.0)

    # Total gas flow reconstructed INSIDE the UKF (algebraically using SPG-O2 from UKF)
    Qg_n = (
        OVL_air +
        OVL_co2 +
        OVL_o2 +
        SPG_air +
        SPG_co2_v[k] +   # known disturbance
        SPG_O2_n         # estimated actuator state
    )

    # OUR is a constraint algebraic state (NO dynamics)
    OUR_n = K_OUR * Qg_n * (yin_n - yout_n)

    return np.array([
        yin_n,      # estimated y_in
        yout_n,     # estimated y_out
        SPG_O2_n,   # estimated effective SPG-O2
        OUR_n,      # estimated OUR (constrained)
        dyin,
        dyout,
        dSPG
    ])

# Measurement model (observation): Feedback, if my system was in state X, what should have my sensor read? 
                                    # Use to predict measurements and compute residuals (innovation)
def hx(x):
    yin, yout, SPG_O2, _, _, _, _ = x
    return np.array([yin, yout, SPG_O2])

dim_z_nodelay = 3

# Sigma points constructor (for Unscented Transform)
points = MerweScaledSigmaPoints(
    n=7,            # State size = same as dim_x (Number of variables of the system) --> it will generate 2n+1 sigma points
    alpha=0.1,      # Spread = Spread of sigma points (how far they should be from the calculated mean) --> higher because of overpeaked distribution of the state
    beta=2.0,       # Shape = Assumption about the distribution shape (2 = good for Gaussian distributions)
    kappa=0.0       # Bias = Extra tuning for sigma points placement
)

# UKF model
ukf = UKF(
    dim_x=7,        # Number of variables of the system (state vector in process model)
    dim_z=dim_z_nodelay,        # Number of variables measured in the sensor (measurement model)
    fx=fx,          # State transition funtion (process model)
    hx=hx,          # Measurement model
    dt=dt,
    points=points   # Sigma points generator
)

# Metrices initialization
OUR0 = K_OUR * Qg[0] * (y_in_raw[0] - y_out[0])

ukf.x = np.array([
    y_in_raw[0],        # yin
    y_out[0],       # yout
    SPG_o2_v[0],    # SPG-O2 (initial actuator state)
    OUR0,           # OUR
    0.0,            # dyin
    0.0,            # dyout
    0.0             # dSPG
])

ukf.P = np.diag([
    1e-5,   # y_in
    1e-5,   # y_out
    1e-4,   # SPG-O2
    5e-2,   # OUR (derived uncertainty)
    1e-7,
    1e-7,
    1e-6
])

# Noise model
ukf.Q = np.diag([
    1e-6,   # y_in
    1e-6,   # y_out
    1e-6,   # SPG-O2 (slow actuator dynamics)
    0.0,    # OUR is constraint, so zero noise
    1e-9,
    1e-9,
    1e-6
])

# Measurement noise
R_yin  = max(np.var(np.diff(y_in_raw)),  1e-8)
R_yout = max(np.var(np.diff(y_out)), 1e-8)
R_SPG  = max(np.var(np.diff(SPG_o2_v)), 1e-6)

alpha_R = 0.03

# Initialize the variables at 0
yin_filt  = np.zeros(len(t_s))
yout_filt = np.zeros(len(t_s))
SPG_filt  = np.zeros(len(t_s))
OUR_filt_v1  = np.zeros(len(t_s))
sigma_OUR_nodelay = np.zeros(len(t_s))
innovation_all_nodelay = np.zeros((len(t_s), dim_z_nodelay))
S_all_nodelay = np.zeros((len(t_s), dim_z_nodelay, dim_z_nodelay))
z_pred_all_nodelay = np.zeros((len(t_s), dim_z_nodelay))
z_meas_all_nodelay = np.zeros((len(t_s), dim_z_nodelay))

# Run the UKF 
for k in range(len(t_s)):
    
     ukf.predict() 
     
     ukf.R = np.diag([ 
         R_yin + alpha_R * ukf.x[0]**2, 
         R_yout + alpha_R * ukf.x[1]**2, 
         R_SPG 
    ]) 
     
     z = np.array([ 
        y_in_raw[k], 
        y_out[k], 
        SPG_o2_v[k] 
    ])

     if np.all(np.isfinite(z)): 
        ukf.update(z)
    
     yin_filt[k]  = ukf.x[0]
     yout_filt[k] = ukf.x[1]
     SPG_filt[k]  = ukf.x[2]
     OUR_filt_v1[k]  = ukf.x[3]
     sigma_OUR_nodelay[k] = np.sqrt(ukf.P[5,5])
     innovation_all_nodelay[k,:] = ukf.y
     S_all_nodelay[k,:,:] = ukf.S
     z_pred_all_nodelay[k,:] = hx(ukf.x)
     z_meas_all_nodelay[k,:] = z

# Store metrics in df
df["OUR_UKF_v1_[mmol/h/L]"] = np.asarray(OUR_filt_v1, dtype=float)

# Metrics UKF with delay
metrics_UKF_NOdelay = evaluate_filter(
    OUR_filt=df["OUR_UKF_v1_[mmol/h/L]"],
    sigma_OUR=sigma_OUR_nodelay,
    innovation=innovation_all_nodelay,
    S_all=S_all_nodelay,
    z_meas=z_meas_all_nodelay,
    z_pred=z_pred_all_nodelay,
    OUR_gas_series=OUR_raw,
    dim_z = dim_z_nodelay 
)

#%%
# 8.3 (v2) UKF FOR RAW INPUTS AND SENSOR DELAY
# ---------------------------------------------------------
# Conversion constant for OUR
K_OUR = 60 * 1000 / (V_l * Vm)

# Time step
dt = max(np.median(np.diff(t_s)), 1e-8)  # hours; avoid numerical issues

SPG_co2_current = 0.0
SPG_o2_current = 0.0

# Initial variables
Qg0 = BASE_FLOW + SPG_co2_v[0] + SPG_o2_v[0]
yin0 = (y_O2_air*(OVL_air + SPG_air) + OVL_o2 + SPG_o2_v[0])/ (Qg0 if Qg0 > 0 else 1e-10)  # avoid division by 0
OUR0 = K_OUR * Qg0 * (yin0 - y_out[0])   

# Process model: Prediction about how the state evolves over time
def fx(x, dt):

    # State vector
    yout, yin_an, yout_an, SPG_O2, SPG_CO2, OUR = x

    # SPG: slow actuator dynamics
    tau_spg = 0.1           # h --> 6 min --> Related to PID controller parameters (Integration Time)

    SPG_O2_n = max(SPG_O2 + (SPG_o2_current - SPG_O2) * dt / tau_spg, 0)        # Positive constraint  
    SPG_CO2_n = max(SPG_CO2 + (SPG_co2_current - SPG_CO2) * dt / tau_spg, 0)    # Positive constraint

    # Total gas flow
    Qg_n = OVL_air + SPG_air + OVL_co2 + SPG_CO2_n + OVL_o2 + SPG_O2_n

    # Oxygen fraction (yin) model
    yin = (y_O2_air*(OVL_air + SPG_air) + OVL_o2 + SPG_O2_n) / Qg_n

    # Analyzer delay: first-order low pass filter
    tau_an = 0.003   # h --> 10.8s --> it will filter out all signals faster than tau 
    yin_an_n  = yin_an  + (yin  - yin_an)  * dt / tau_an
    yout_an_n = yout_an + (yout - yout_an) * dt / tau_an

    # OUR: soft physics + random walk
    OUR_model = K_OUR * Qg_n * (yin_an_n - yout_an_n)
    tau_OUR = 0.5       # h --> 30 min --> good to keep fast bio dynamics but not sensitive to noise yet
    OUR_n = OUR + (OUR_model - OUR) * dt / tau_OUR 

    return np.array([yout, yin_an_n, yout_an_n, SPG_O2_n, SPG_CO2_n, OUR_n])

# Measurement model (observation): Feedback, if my system was in state X, what should have my sensor read? 
                                    # Use to predict measurements and compute residuals (innovation)
def hx(x):
    yout, yin_an, yout_an, SPG_O2, SPG_CO2, _ = x    # What informs the measurements
    return np.array([yout_an, SPG_O2, SPG_CO2])      # What I really measure

# State the measurement vector dimension
dim_z = 3

# Sigma points constructor (for Unscented Transform)
points = MerweScaledSigmaPoints(
    n=6,
    alpha=0.1,     # Non-linear spread
    beta=2.0,
    kappa=0.0
)

# UKF model
ukf = UKF(
    dim_x=6,
    dim_z=dim_z,
    fx=fx,
    hx=hx,
    dt=dt,
    points=points
)

ukf.x = np.array([
    y_out[0],        # yout
    yin0,            # yin_an: analyzer starts aligned
    y_out[0],        # yout_an: analyzer starts aligned
    SPG_o2_v[0],     # SPG-O2
    SPG_co2_v[0],    # SPG-CO2
    OUR0             # OUR
])

# Uncertainty model (covariance matrix)
ukf.P = np.diag([
    1e-5,   # yout
    5e-5,   # yin_an
    5e-5,   # yout_an
    2e-4,   # SPG-O2
    2e-4,   # SPG-CO2
    5e-2    # OUR --> allow freedom to random walk
])

# Process noise model
ukf.Q = np.diag([
    1e-6,           # yout
    1e-6,           # yin_an; intial noise, updated afterwads
    1e-7,           # yout_an: analyzer states low noise
    1e-6,           # SPG-O2; min noise + real noise
    1e-6,           # SPG-CO2; min noise + real noise
    1e-4            # OUR: initial noise, updated afterwards
])

# Uncertainty contributions from input flows
A = y_O2_air*(OVL_air + SPG_air) + OVL_o2 + SPG_o2_v

sigma_A = np.sqrt((y_O2_air**2)*(sigma_OVL_air**2 + sigma_SPG_air**2) + sigma_OVL_o2**2 + sigma_SPG_o2**2)

# Covariance due to shared SPG_O2_v
cov_A_Qg = sigma_SPG_o2**2

sigma_yin = np.sqrt(
    (sigma_A / Qg)**2 +
    ((A * sigma_Qg) / (Qg**2))**2 -2*(A/Qg**3)*cov_A_Qg
)

# Initialize the variables at 0
yin_filt  = np.zeros(len(t_s))
yout_filt = np.zeros(len(t_s))
SPG_filt  = np.zeros(len(t_s))
OUR_filt  = np.zeros(len(t_s))
sigma_OUR = np.zeros(len(t_s))
innovation_all = np.zeros((len(t_s), dim_z))
S_all = np.zeros((len(t_s), dim_z, dim_z))
z_pred_all = np.zeros((len(t_s), dim_z))
z_meas_all = np.zeros((len(t_s), dim_z))

rel_error = 0.03  # 3% relative

# Run the UKF 
for k in range(len(t_s)):

    # 0. SPG information update
     SPG_o2_current = SPG_o2_v[k]
     SPG_co2_current = SPG_co2_v[k]

     ukf.Q[1,1] = sigma_yin[k]**2

    # 1. Adaptive OUR process noise                       
     delta_y = yin_filt[k-1] - yout_filt[k-1] if k > 0 else (yin0 - y_out[0])          # yin_an - yout_an approx
     
     sigma_OUR_dy = np.sqrt(       # Extra uncertainty of the constants that are not propagated by sigma points
        (K_OUR * Qg[k])**2 * sigma_yO2out[k]**2 +    
        (K_OUR * delta_y)**2 * sigma_Qg**2
     )
     ukf.Q[5,5] = 1e-4 + sigma_OUR_dy[k]**2  # base OUR process noise + sigma

    # 2. Predict step
     ukf.predict()     

    # 4. Measurement noise covariance matrix (how much to trust the sensors)
     ukf.R = np.diag([ 
        max(sigma_yO2out[k]**2 + rel_error*ukf.x[0]**2, 1e-4),          # minimum noise to prevent overconfidence
        max(sigma_SPG_o2[k]**2 + rel_error*ukf.x[3]**2, 1e-4),          # minimum noise to prevent overconfidence
        max(sigma_SPG_co2[k]**2 + rel_error*ukf.x[4]**2, 1e-4)          # minimum noise to prevent overconfidence
    ])  

    # 5. Measurement vector
     z = np.array([ 
        y_out[k], 
        SPG_o2_v[k],
        SPG_co2[k] 
     ])

    # 6. Update step
     if np.all(np.isfinite(z)): 
        ukf.update(z)

     yin_filt[k]  = ukf.x[1]        # hidden state
     yout_filt[k] = ukf.x[0]
     SPG_filt[k]  = ukf.x[3]
     OUR_filt[k]  = ukf.x[5]        # hidden state
     sigma_OUR[k] = np.sqrt(ukf.P[5,5])
     innovation_all[k,:] = ukf.y
     S_all[k,:,:] = ukf.S
     z_pred_all[k,:] = hx(ukf.x)
     z_meas_all[k,:] = z

# Store metrics in df
df["y_in_UKF_v2_[perc-vol]"]  = yin_filt
df["y_out_UKF_v2_[perc-vol]"] = yout_filt
df["SPG_O2_UKF_v2_[slpm]"]    = SPG_filt
df["OUR_UKF_v2_[mmol/h/L]"]   = np.asarray(OUR_filt, dtype=float)
df["sigma_OUR_v2_[mmol/h/L]"] = sigma_OUR        # Uncertainty propagated by UKF

# Metrics UKF with delay
metrics_UKF_v2 = evaluate_filter(
    OUR_filt=df["OUR_UKF_v2_[mmol/h/L]"],
    sigma_OUR=df["sigma_OUR_v2_[mmol/h/L]"],
    innovation=innovation_all,
    S_all=S_all,
    z_meas=z_meas_all,
    z_pred=z_pred_all,
    OUR_gas_series=OUR_raw,
    dim_z = dim_z
)

# %%
# 8.4 UKF TEST AND COMPARISON
# ---------------------------------------------------------
# IMPORTANT
# In my UKF, I assumed my system is close to Gaussian (beta=2.0 in sigma points generation)
# I have to double check that:

res_ukf_v2 = OUR_raw - df["OUR_UKF_v2_[mmol/h/L]"]          # Residuals
#res_ukf_v1 = OUR_raw - df["OUR_UKF_v1_[mmol/h/L]"]
#res_ukf_v0 = OUR_raw -df["OUR_v0_[mmol/h/L]"]

#1 Mean of residuals (innovation) should be close to 0 --> true?
mu = np.mean(res_ukf_v2, axis=0)
var = np.var(res_ukf_v2, axis=0)

print("-------------------------------------------")
print("CONDITION 1: The Mean should be close to 0:")
print(">Mean innovations =", mu)
print(">Variance innovations =", var)
print(">Std innovations =", np.std(res_ukf_v2))

# Plot innovation: 
plt.figure(figsize=(8,5))
plt.plot(t_s, res_ukf_v2, alpha = 0.5, label = "Residuals OUR (UKF)")
plt.axhline(y=0, color='black', linestyle=':', label = "y = 0")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
#plt.title("Innovation")
plt.xlabel("Time [h]")
plt.legend()
plt.ylabel("Residuals [mmol/h/L]")
plt.show()

#2 Innovations should be white (uncorrelated)
# We already checked that with the PSD --> the high frequency (noise) was flat --> white noise
print("-------------------------------------------")
print("CONDITON 2: The noise should be white (uncorrelated)")
print("Using the Power Spectral Density (PSD)")
fs=1/dt

f_ukf, Pxx_ukf = welch(OUR_raw, fs=fs)
f_ukf_sg, Pxx_ukf_sg = welch(OUR_SG, fs=fs)
f_ukf_v1, Pxx_ukf_v1 = welch(df["OUR_UKF_v1_[mmol/h/L]"].to_numpy().ravel(), fs=fs)
f_ukf_v2, Pxx_ukf_v2 = welch(df["OUR_UKF_v2_[mmol/h/L]"].to_numpy().ravel(), fs=fs)
f_ukf_v0, Pxx_ukf_v0 = welch(df["OUR_v0_[mmol/h/L]"].to_numpy().ravel(), fs=fs)

plt.figure(figsize=(5,5))
plt.semilogy(f_ukf, Pxx_ukf, label="Raw", alpha = 0.8)
plt.semilogy(f_ukf_sg, Pxx_ukf_sg, label="SG",alpha = 0.8, color = "red")
plt.semilogy(f_ukf_v0, Pxx_ukf_v0, label="UKF v0",alpha = 0.8)
plt.semilogy(f_ukf_v1, Pxx_ukf_v1, label="UKF v1",alpha = 0.8)
plt.semilogy(f_ukf_v2, Pxx_ukf_v2, label="UKF v2", alpha = 0.8, color = "purple")
plt.axvline(x=30, color='grey', linestyle=':', label = "30 1/h")
plt.yscale("log")
plt.ylabel("Power Spectral Density (PSD) [(mmol/h/L)² h]")
plt.xlabel("f [1/h]")
plt.xlim(75, 100)
plt.axvline(x=89)
plt.legend()
#plt.title("Power spectrum of OUR extracted trend")
plt.show()

print("Can be observed in the high frequency innovations --> flat")

#3 Should follow a Normal Distribution
print("-------------------------------------------")
print("CONDITON 3: The system should follow a Gaussian distribution")
data = res_ukf_v2
sigma = np.sqrt(var)

# Assuming mu and sigma are defined for Gaussian
# For Laplace, we'll estimate the location (loc) and scale (b)
loc = np.mean(res_ukf_v2)         # Laplace location parameter
b = np.mean(np.abs(res_ukf_v2 - loc))  # Laplace scale parameter

plt.figure(figsize=(5,5))
plt.hist(res_ukf_v2, bins=150, density=True, alpha=0.5, label="OUR UKF")

x = np.linspace(data.min(), data.max(), 300)
plt.xlim(-75,75)

# Gaussian PDF
plt.plot(x, norm.pdf(x, mu, sigma), label="Gaussian")

# Laplace PDF
plt.plot(x, laplace.pdf(x, loc, b), label="Laplace")

plt.xlabel("Residual")
plt.ylabel("Probability density")
#plt.title("Residual distribution")
plt.legend()
plt.show()

print("It does not, but it is good enough for a Kalman Filter (no presence of long tails)")
print("Just increase a bit beta for the sigma points --> better for overpeaking")

#Plot: Compare different UKFs
plt.figure(figsize=(8,5))
#plt.plot(t_s, OUR_raw, alpha=0.5, label="OUR raw", zorder=0)
plt.plot(t_s, OUR_SG, alpha = 0.6, label="SG", color = "red", zorder=3)
plt.plot(t_s, df["OUR_v0_[mmol/h/L]"], alpha = 1, label="UKF v0", color="orange", zorder=4)
plt.plot(t_s, df["OUR_UKF_v1_[mmol/h/L]"], alpha = 0.5, label="UKF v1", color ="green", zorder=2)
plt.plot(t_s, df["OUR_UKF_v2_[mmol/h/L]"], alpha = 0.6, label="UKF v2", color = "purple", zorder=5)

#plt.xlim(155,156)
#plt.ylim(5,12)

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
#plt.axvspan(104, 118, color='grey', alpha=0.2, lw=0)
#plt.axvspan(122, 142, color='grey', alpha=0.2, lw=0)
#plt.axvspan(210, 240, color='grey', alpha=0.2, lw=0)

plt.xlabel("Time [h]")
plt.ylabel("OUR [mmol/h·L]")
plt.legend()
plt.show()

# Plot OUR (v2) +- sigma
plt.figure(figsize=(6,5))
plt.plot(t_s, df["OUR_UKF_v2_[mmol/h/L]"], label = "OUR", alpha = 0.8)
plt.fill_between(t_s, df["OUR_UKF_v2_[mmol/h/L]"] + df["sigma_OUR_v2_[mmol/h/L]"], df["OUR_UKF_v2_[mmol/h/L]"] - df["sigma_OUR_v2_[mmol/h/L]"], label = "σ", alpha = 0.3, color="#24B8D6")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
plt.ylabel("OUR [mmol/h·L]")
plt.legend()
plt.title("")
plt.show()

# %%
# 9. CALCULATE qO2 from OUR UKF (v2)
# ---------------------------------------------------------
# OUR = qO2 * VCD
# Since qO2 = OUR / VCD and, at the beginning, VCD = 0  

# Compute average OUR per minute (to fit measurement timescale for VSD)
OUR_raw_min = (             # [mmol/h/L]
    df.assign(t_s=pd.to_timedelta(df["t_[s]"], unit="s"))
      .set_index("t_s")["OUR_raw_[mmol/h/L]"]       #VERY IMPORTANT: choose the OUR to use FROM df (raw, smooth_SG, smooth_KF, etc.)
      .resample("1min")
      .mean()
)

OUR_min = (             # [mmol/h/L]
    df.assign(t_s=pd.to_timedelta(df["t_[s]"], unit="s"))
      .set_index("t_s")["OUR_UKF_v2_[mmol/h/L]"]       #VERY IMPORTANT: choose the OUR to use FROM df (raw, smooth_SG, smooth_KF, etc.)
      .resample("1min")
      .mean()
)

sigma_OUR_min = (                           # Downsampling of sigma OUR (before performed with OUR as well)
    df.assign(t_s=pd.to_timedelta(df["t_[s]"], unit="s"))
      .set_index("t_s")["sigma_OUR_v2_[mmol/h/L]"]
      .resample("1min")
      .apply(lambda x: np.sqrt(np.mean(x**2)))
)

# Calculate sigma for qO2 
OUR_safe = np.maximum(OUR_min.to_numpy(), 1e-6)    # No division by 0

# Calculate qO2
qO2 = (OUR_safe* 1e9 * 24) / VCD_safe   # [pmol/vc/d]

# qO2 smooth for subsequent calculations (the resolution of metabolites is very low, no high detail needed)
w_qO2 = 721   # must be odd → ~ 12h window
p_qO2 = 2

qO2_smooth = savgol_filter(qO2, window_length=w_qO2, polyorder=p_qO2)

# Relative uncertainties
rel_OUR = sigma_OUR_min / OUR_safe
rel_VCD = sigma_VCD_min / VCD_safe

# Propagation (dependent case)
rho = np.corrcoef(OUR_raw_min, VCD)[0,1]        # Correlation (dependency) of OUR and VCD
sigma_qO2 = qO2 * np.sqrt(rel_OUR**2 + rel_VCD**2 - 2*rho*rel_OUR*rel_VCD)

# Plot qO2 +- sigma
plt.figure(figsize=(5,5))
plt.fill_between(t_vcd, np.maximum(qO2 - sigma_qO2,0), np.minimum(qO2 + sigma_qO2,250), alpha=0.3, color="green", label="σ qO₂")
plt.plot(t_vcd, qO2, color="green", label="qO₂")
plt.ylabel("qO₂ [pmol/vc/d]")
plt.xlabel("Time [h]")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.legend()
plt.show()

# %%
# 10.1 LOAD OFFLINE-MEASURED METABOLITES
# -------------------------------------------
metabolites = pd.read_csv("path\\to\\file\\metabolites.csv", 
                          sep=';', decimal=',', parse_dates=["date"], dayfirst=True)

# Strip whitespace from column names
metabolites.columns = metabolites.columns.str.strip()

#Extract date
metabolites["time"] = pd.to_datetime(metabolites["date"], format="%Y-%m-%d %H:%M")
metabolites["t_[h]"] = (metabolites["time"] - t0).dt.total_seconds() / 3600     # Create t_[h]
metabolites = metabolites.sort_values("t_[h]")  # Sort time ascending

# Transform to mmol/L
Mw_Gluc = 180.156   # [g/mol]
Mw_Lac = 90.078     # [g/mol]

metabolites["c(Gluc)_[mmol/L]"] = round(metabolites["c(Gluc)_[g/L]"]/Mw_Gluc*1000, 4)
metabolites["c(Lac)_[mmol/L]"] = round(metabolites["c(Lac)_[g/L]"]/Mw_Lac*1000, 4)

metabolites["qGluc_[pmol/vc/d]"] = round(metabolites["qGluc_[ng/c/d]"]/Mw_Gluc*1000, 4)
metabolites["qLac_[pmol/vc/d]"] = round(metabolites["qLac_[ng/c/d]"]/Mw_Lac*1000, 4)

# Plot Gluc and Lac
plt.figure(figsize=(5,5))
plt.scatter(metabolites["t_[s]"]/3600, metabolites["c(Lac)_[mmol/L]"], label = "Lactate")
plt.plot(metabolites["t_[s]"]/3600, metabolites["c(Lac)_[mmol/L]"], alpha= 0.3)
plt.scatter(metabolites["t_[s]"]/3600, metabolites["c(Gluc)_[mmol/L]"], label = "Glucose")
plt.plot(metabolites["t_[s]"]/3600, metabolites["c(Gluc)_[mmol/L]"], alpha = 0.3)
plt.ylabel("Concentration [mmol/L]")
plt.xlim(-10,305)
plt.xlabel("Time [h]")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.legend()
plt.show()

# %%
# 10.2 qS CALCULATION
# -------------------------------------------
# Load variables
t_met = metabolites["t_[h]"].to_numpy()

Gluc = metabolites["c(Gluc)_[mmol/L]"].to_numpy()
Lac  = metabolites["c(Lac)_[mmol/L]"].to_numpy()

# Uncertainty glucose and lactate
sigmaB_gluc = np.maximum(0.07*Gluc, 0.07)/np.sqrt(3)     # From Flex2 Installation and Operational Qualification (IQOQ) Document
sigmaB_lac  = np.maximum(0.07*Lac, 0.07)/np.sqrt(3)      # From Flex2 Installation and Operational Qualification (IQOQ) Document

sigma_gluc = np.sqrt(sigmaA_gluc**2+sigmaB_gluc**2)
sigma_lac = np.sqrt(sigmaA_lac**2+sigmaB_lac**2)

# Interpolate states at metabolite times
VCD_met = np.interp(t_met, vcd_df["t_[h]"], vcd_df["VCD_[E6/ml]"]) * 1e9       # cells/L
D_met = np.interp(t_met, weight["t_[h]"], D)                                   # 1/h--> RV/h
qO2_met = np.interp(t_met, t_vcd, qO2)                                         # pmol/vc/d

# Interpolate uncertainties at metabolite times
sigma_VCD_met = np.interp(t_met, vcd_df["t_[h]"], sigma_VCD_min) * 1e9       # cells/L
sigma_D_met = np.interp(t_met, weight["t_[h]"], sigma_D)                                   # 1/h--> RV/h
sigma_qO2_met = np.interp(t_met, t_vcd, sigma_qO2)                                         # pmol/vc/d

# Calculate dS
dGluc_dt = np.gradient(Gluc, t_met)     # mmol/h
dLac_dt = np.gradient(Lac, t_met)       # mmol/h

# Feed concentrations
Sin_gluc_gl = 5     # g/L
Sin_gluc = Sin_gluc_gl / Mw_Gluc * 1000     # mmol/L
Sin_lac  = 0.0

# qS calculation from perfusion balance
qGluc = - (dGluc_dt - D_met * (Sin_gluc - Gluc)) / VCD_met    # mmol/vc/h, (negative for consumption)
qLac  = (dLac_dt  - D_met * (Sin_lac  - Lac )) / VCD_met    # mmol/vc/h

# Convert to pmol/cell/d
qGluc *= 1e9 * 24
qLac  *= 1e9 * 24

metabolites["qGluc_calc_[pmol/c/d]"] = qGluc
metabolites["qLac_calc_[pmol/c/d]"]  = qLac

# Calculate metabolic yield coefficient (Y)
Y_Lac_Gluc = qLac / qGluc     
#Y_O2_Gluc = - qO2_met / qGluc   # [pmol/vc/d] / [pmol/vc/d] ; How much oxygen is consumed per glucose consumed

metabolites["Y_Lac_Gluc_[]"] = Y_Lac_Gluc
#metabolites["Y_O2_Gluc_[]"] = Y_O2_Gluc

# Monte Carlo uncertainty propagation
N_mc_S = 5000
qGluc_mc = np.empty((N_mc_S, len(qGluc)))
qLac_mc  = np.empty((N_mc_S, len(qLac)))
Y_Lac_Gluc_mc = np.empty((N_mc_S, len(qGluc)))
Y_O2_Gluc_mc  = np.empty((N_mc_S, len(qGluc)))

for i in range(N_mc_S):
    # Get perturbed variables
    Gluc_i = Gluc + rng.normal(0, sigma_gluc)
    Lac_i  = Lac  + rng.normal(0, sigma_lac)
    D_i    = D_met + rng.normal(0, sigma_D_met)
    VCD_i  = VCD_met + rng.normal(0, sigma_VCD_met)
    qO2_i  = qO2_met + rng.normal(0, sigma_qO2_met)

    # Calculate qS
    dGluc_dt_i = np.gradient(Gluc_i, t_met)
    dLac_dt_i  = np.gradient(Lac_i, t_met)

    qGluc_i = - (dGluc_dt_i - D_i*(Sin_gluc - Gluc_i)) / VCD_i * 1e9 * 24   # Negative for consumption
    qLac_i  = (dLac_dt_i  - D_i*(Sin_lac - Lac_i)) / VCD_i * 1e9 * 24

    qGluc_mc[i] = qGluc_i
    qLac_mc[i]  = qLac_i

    # Calculate yields for this iteration
    Y_Lac_Gluc_mc[i] = - qLac_i / qGluc_i
    Y_O2_Gluc_mc[i]  = - qO2_i / qGluc_i

# Mean and std for rates
qGluc_mean  = np.mean(qGluc_mc, axis=0)
sigma_qGluc = np.std(qGluc_mc, axis=0)
qLac_mean   = np.mean(qLac_mc, axis=0)
sigma_qLac  = np.std(qLac_mc, axis=0)

# Mean and std for yields
Y_Lac_Gluc_mean  = np.mean(Y_Lac_Gluc_mc, axis=0)
sigma_Y_Lac_Gluc = np.std(Y_Lac_Gluc_mc, axis=0)
Y_O2_Gluc_mean   = np.mean(Y_O2_Gluc_mc, axis=0)
sigma_Y_O2_Gluc  = np.std(Y_O2_Gluc_mc, axis=0)

# Plot: qS  
plt.figure(figsize=(5,5))
plt.plot(t_met, qLac_mean, alpha= 0.3)
plt.scatter(t_met, qLac_mean, label = "qLac (production)", zorder=10)
plt.fill_between(t_met, qLac_mean + sigma_qLac, qLac_mean - sigma_qLac, color = "tab:blue", alpha=0.3)
plt.plot(t_met, qGluc_mean, alpha= 0.3)
plt.scatter(t_met, qGluc_mean, label = "qGluc (consumption)", zorder=10)
plt.fill_between(t_met, qGluc_mean + sigma_qGluc, qGluc_mean - sigma_qGluc, color = "orange", alpha=0.3)
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
#plt.ylim(-10,20)
plt.xlim(-10,305)
plt.legend()
plt.ylabel("qS [pmol/vc/d]")
plt.show()

# Plot: Y(Lac/Gluc) 
plt.figure(figsize=(5,5))
plt.plot(t_met, Y_Lac_Gluc, color = "purple", alpha=0.3)
plt.scatter(t_met, Y_Lac_Gluc, label = "Y(Lac/Gluc)", color = "purple")
#plt.fill_between(t_met, Y_Lac_Gluc + sigma_Y_Lac_Gluc, Y_Lac_Gluc - sigma_Y_Lac_Gluc, color = "green", alpha=0.3)
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
plt.xlim(-10,305)
plt.legend()
plt.ylabel("Yield [-]")
plt.show()

# Plot qO2, qGlu and qLac:
fig, ax1 = plt.subplots(figsize=(5,5))

ax1.plot(t_vcd, qO2_smooth, label=f"qO₂", color = "green", alpha = 0.8)
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("qO₂ [pmol/vc/d]")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
plt.plot(t_met, qLac_mean, alpha= 0.3)
plt.scatter(t_met, qLac_mean, label = "qLac", zorder=10)
plt.plot(t_met, qGluc_mean, alpha= 0.3)
plt.scatter(t_met, qGluc_mean, label = "qGluc", zorder=10)
ax2.set_ylabel("qGluc, qLac [pmol/vc/d]")
ax2.tick_params(axis='y')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc=1)

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.title("")
plt.xlabel("Time [h]")

plt.show()

# Plot qO2 and Y(O2/Gluc):
fig, ax1 = plt.subplots(figsize=(5,5))

ax1.plot(t_vcd, qO2_smooth, label=f"qO₂", color = "green", alpha = 0.8)
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("qO₂ [pmol/vc/d]")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.plot(t_met, Y_Lac_Gluc, color = "purple", alpha=0.3)
ax2.scatter(t_met, Y_Lac_Gluc, label = "Y(Lac/Gluc)", color = "purple")
ax2.set_ylabel("Yield [-]")
ax2.tick_params(axis='y')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.title("")
plt.xlabel("Time [h]")
plt.show()

# %%
# 10.3 FURTHER METABOLITE ANALYSIS
# -------------------------------------------
# Get metabolites
Glu = metabolites["c(Glu)_[mmol/L]"].to_numpy()
Gln  = metabolites["c(Gln)_[mmol/L]"].to_numpy()
NH4  = metabolites["c(NH4+)_[mmol/L]"].to_numpy()

FVIII= metabolites["FVIII-HLP_[mg/ml]"].to_numpy()*1000     # mg/l
DNAt = metabolites["DNA_t_[ng/ul]"].to_numpy()              # mg/l (same)
HCP = metabolites["HCP_[ng/ml]"].to_numpy()*0.001           # mg/l

# Calculate dS
dGlu_dt = np.gradient(Glu, t_met)       # mmol/h
dGln_dt = np.gradient(Gln, t_met)       # mmol/h
dNH4_dt = np.gradient(NH4, t_met)       # mmol/h

dFVIII_dt = np.gradient(FVIII, t_met)      # mg/h
dDNAt_dt = np.gradient(DNAt, t_met)        # mg/h
dHCP_dt = np.gradient(HCP, t_met)          # mg/h

# Molecular weights
Mw_Glu = 147.13     # g/mol
Mw_Gln = 146.15     # g/mol
Mw_Ala = 89.09      # g/mol

# Feed concentrations
Sin_Glu_mgl = 209     # mg/L
Sin_Glu = Sin_Glu_mgl / Mw_Glu     # mmol/L

Sin_Gln_Ala_mgl = 869   # mg/L
Sin_Gln = Sin_Gln_Ala_mgl / (Mw_Gln+Mw_Ala)    # mmol/L

Sin_NH4  = 0.0
Sin_FVIII  = 0.0
Sin_DNAt  = 0.0
Sin_HCP  = 0.0

# qS calculation from perfusion balance
qGlu = (dGlu_dt - D_met * (Sin_Glu - Glu)) / VCD_met        # mmol/vc/h
qGln  = (dGln_dt  - D_met * (Sin_Gln  - Gln )) / VCD_met    # mmol/vc/h
qNH4  = (dNH4_dt  - D_met * (Sin_NH4  - NH4 )) / VCD_met    # mmol/vc/h

qFVIII  = (dFVIII_dt  - D_met * (Sin_FVIII  - FVIII)) / VCD_met    # mg/vc/h
qDNAt  = (dDNAt_dt  - D_met * (Sin_DNAt  - DNAt)) / VCD_met        # mg/vc/h
qHCP  = (dHCP_dt  - D_met * (Sin_HCP  - HCP)) / VCD_met            # mg/vc/h

# Convert to pmol/cell/d
qGlu *= -1e9 * 24        # pmol/cell/d; consumption
qGln *= -1e9 * 24        # pmol/cell/d; consumption
qNH4 *= 1e9 * 24        # pmol/cell/d

qFVIII *= 1e9 * 24       # pg/cell/d
qDNAt  *= 1e9 * 24       # pg/cell/d
qHCP  *= 1e9 * 24        # pg/cell/d

# Plot qO2, qGlu, qGln and qNH4+:
fig, ax1 = plt.subplots(figsize=(5,5))

ax1.plot(t_vcd, qO2_smooth, label=f"qO₂", color = "grey", zorder=0, alpha = 0.8)
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("qO₂ [pmol/vc/d]")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
plt.plot(t_met, qGlu, alpha= 0.3, color = "green")
plt.scatter(t_met, qGlu, label = "qGlu", zorder=10,  color ="green")
plt.plot(t_met, qGln, alpha= 0.3)
plt.scatter(t_met, qGln, label = "qGln", zorder=10)
plt.plot(t_met, qNH4, alpha= 0.3, color = "purple")
plt.scatter(t_met, qNH4, label = "qNH4", zorder=10, color = "purple")
ax2.set_ylabel("qS [pmol/vc/d]")
ax2.tick_params(axis='y')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc=4)

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.title("")
plt.xlabel("Time [h]")
plt.show()

# Plot qO2, qFVIII, qDNAt and qHCP:
fig, ax1 = plt.subplots(figsize=(5,5))

ax1.plot(t_vcd, qO2_smooth, label=f"qO₂", color = "grey", zorder=0, alpha = 0.8)
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("qO₂ [pmol/vc/d]")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
plt.plot(t_met, qFVIII, alpha= 0.3, color = "red")
plt.scatter(t_met, qFVIII, label = "qFVIII", zorder=10, color = "red")
plt.plot(t_met, qDNAt, alpha= 0.3, color = "orange")
plt.scatter(t_met, qDNAt, label = "qDNAt", zorder=10, color = "orange")
#plt.plot(t_met, qHCP, alpha= 0.3)
#plt.scatter(t_met, qHCP, label = "qHCP", zorder=10)
ax2.set_ylabel("qS [pg/vc/d]")
ax2.tick_params(axis='y')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc=4)

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.title("")
plt.xlabel("Time [h]")
plt.show()

#%%
# 11. LOOK FOR qO2 qGluc AND qLac RELATIONSHIPS
# ---------------------------------------------------------
# Function for local qO2 averaging (downsampling)
def compute_local_qO2_avg(t_low, t_high):
    # Determine the range
    t_low = max(t_low, t_vcd[0])
    t_high = min(t_high, t_vcd[-1])

    if t_high <= t_low:     # Security check: range is correct
        return np.nan

    # Select the time window
    mask = (t_vcd >= t_low) & (t_vcd <= t_high)
    t_window = t_vcd[mask]
    q_window = qO2[mask]

    if len(q_window) < 2:
        return np.nan       # Security check: need at least 2 points for trapezoid integration

    # Compute AUC using trapezoid integration
    auc = np.trapezoid(q_window, t_window)

    # Average over the interval length
    avg = auc / (t_window[-1] - t_window[0])
    return avg

# Compute local averages for each metabolite time
qO2_avg_local = []
for i in range(len(t_met)):
    if i == 0:
        t_low = 0
        t_high = t_met[i+1]
    elif i == len(t_met)-1:
        t_low = t_met[i-1]
        t_high = t_vcd[-1]
    else:
        t_low = t_met[i-1]
        t_high = t_met[i+1]

    avg = compute_local_qO2_avg(t_low, t_high)
    qO2_avg_local.append(avg)

metabolites["qO2_[pmol/vc/d]_avg"] = qO2_avg_local

# Indicate ramp-up and production phases
phases = {"Ramp-up": t_met <= t_prod0_h, "Production": t_met > t_prod0_h}

# Variables to test
variables = {
    "qLac": "qLac_calc_[pmol/c/d]",
    "qGluc": "qGluc_calc_[pmol/c/d]",
    "Yield": "Y_Lac_Gluc_[]"
}

metrics = {}
summary_rows = []

for phase_name, phase_mask in phases.items():
    print(f"==============================")
    print(f"    {phase_name} Phase      ")
    print(f"==============================")
    
    # Subset data per phase (ramp-up & production)
    qO2_phase = metabolites["qO2_[pmol/vc/d]_avg"][phase_mask].to_numpy()
    qLac_phase = metabolites["qLac_calc_[pmol/c/d]"][phase_mask].to_numpy()
    qGluc_phase = metabolites["qGluc_calc_[pmol/c/d]"][phase_mask].to_numpy()
    yield_phase = metabolites["Y_Lac_Gluc_[]"][phase_mask].to_numpy()

    phase_data = {
        "qLac [pmol/vc/d]": qLac_phase,
        "qGluc [pmol/vc/d]": qGluc_phase,
        "Y(Lac/Gluc) []": yield_phase
    }

    metrics[phase_name] = {}

    # Perform regressions
    for var_name, values in phase_data.items():
        print(f"\n>>>>> {phase_name} - {var_name} vs qO2 <<<<<")
        print("------------------------------------------------\n")

        # Pearson correlation
        #r, p = pearsonr(values, qO2_phase)
        #print(f"Pearson correlation (linear): r = {r:.3f}, p = {p:.3f}")
        
        #metrics[phase_name][var_name] = {"pearson_r": r, "pearson_p": p}

        # Prepare qO2 (X) and other qS values (y) for the incoming regressions
        X = qO2_phase.reshape(-1, 1)
        y = values
        n = len(y)

        loo = LeaveOneOut()
        
        model_metrics = {}

        # Initialize combined plot
        plt.figure()
        plt.scatter(X, y, color="purple", label="Data") # #F8833F

        # FLAT MODEL (MEAN BASELINE)
        # =========================
        y_mean = np.mean(y)
        y_pred_mean = np.full_like(y, y_mean)

        # Metrics
        mean_r2 = r2_score(y, y_pred_mean)
        mean_rmse = np.sqrt(mean_squared_error(y, y_pred_mean))

        # LOOCV for mean (important: recompute mean each time)
        y_pred_cv = np.zeros(len(y))
        for train_idx, test_idx in loo.split(y):
            y_train = y[train_idx]
            y_pred_cv[test_idx] = np.mean(y_train)

        mean_rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))

        # AICc calculation
        def compute_aicc(n, rss, k):
            if n - k - 1 <= 0:
                return np.inf
            aic = n * np.log(rss / n) + 2 * k
            return aic + (2 * k * (k + 1)) / (n - k - 1)

        rss_mean = np.sum((y - y_pred_mean)**2)
        aicc_mean = compute_aicc(n, rss_mean, k=1)  # 1 parameter (mean)

        print(f" FLAT MODEL (mean): y =", round(y_mean,4))
        print(f"    Statistics: R2 = {mean_r2:.4f}, RMSE (train) = {mean_rmse:.4f}, RMSE (LOOCV) = {mean_rmse_cv:.4f}, AICc = {aicc_mean:.4f}\n")

        # Plot
        plt.axhline(y_mean, color="grey", linestyle=":", alpha=0.8, label="Mean")

        model_metrics["Flat model"] = (mean_r2, mean_rmse, mean_rmse_cv, aicc_mean)

        # LINEAR REGRESSION: only if |r| > threshold
        # =========================
        # Model
        lin_model = LinearRegression()
        lin_model.fit(X, y)
        lin_pred = lin_model.predict(X)

        # Statistics
        lin_r2 = r2_score(y, lin_pred)  # R2
        lin_rmse = np.sqrt(mean_squared_error(y, lin_pred)) # RMSE

        y_pred_cv = np.zeros(len(y))                    # LOOCV
        for train_idx, test_idx in loo.split(X):
            lin_model.fit(X[train_idx], y[train_idx])
            y_pred_cv[test_idx] = lin_model.predict(X[test_idx])

        lin_rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))

        rss_lin = np.sum((y - lin_pred)**2)
        aicc_lin = compute_aicc(n, rss_lin, k=2)

        model_metrics["Linear"] = (lin_r2, lin_rmse, lin_rmse_cv, aicc_lin)

        print(f" LINEAR MODEL: coef = {lin_model.coef_[0]:.4f}, intercept = {lin_model.intercept_:.4f}")
        print(f"    Statistics: R2 = {lin_r2:.4f}, RMSE (train) = {lin_rmse:.4f}, RMSE (LOOCV) = {lin_rmse_cv:.4f}\n")

        # Plot
        x_sorted_idx = np.argsort(X[:,0])
        plt.plot(X[x_sorted_idx], lin_pred[x_sorted_idx], linestyle="--", color="#D0593E", alpha=0.8, label="Linear")

        # QUADRATIC MODEL
        # =========================
        poly = PolynomialFeatures(degree=2)
        X_quad = poly.fit_transform(X)

        quad_model = LinearRegression()
        quad_model.fit(X_quad, y)
        quad_pred = quad_model.predict(X_quad)

        quad_r2 = r2_score(y, quad_pred)
        quad_rmse = np.sqrt(mean_squared_error(y, quad_pred))

        y_pred_cv = np.zeros(len(y))                # LOOCV
        for train_idx, test_idx in loo.split(X_quad):
            quad_model.fit(X_quad[train_idx], y[train_idx])
            y_pred_cv[test_idx] = quad_model.predict(X_quad[test_idx])

        quad_rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))

        rss_quad = np.sum((y - quad_pred)**2)
        aicc_quad = compute_aicc(n, rss_quad, k=3)

        model_metrics["Quadratic"] = (quad_r2, quad_rmse, quad_rmse_cv, aicc_quad)

        b0 = quad_model.intercept_
        b1 = quad_model.coef_[1]
        b2 = quad_model.coef_[2]

        print(f" QUADRATIC MODEL: y = {b0:.4f} + {b1:.4f}*x + {b2:.4f}*x^2")
        print(f"    Statistics: R2 = {quad_r2:.4f}, RMSE (train) = {quad_rmse:.4f}, RMSE (LOOCV) = {quad_rmse_cv:.4f}\n")

        x_vals = np.linspace(X.min(), X.max(), 300)
        y_vals = b0 + b1*x_vals + b2*(x_vals**2)

        # Plot
        plt.plot(x_vals, y_vals, linestyle="-.", color="#D0A13E", alpha=0.8, label="Quadratic")

        # LOG MODEL (always if X > 0)
        # =========================
        if np.all(X > 0):
            X_log = np.log(X)

            log_model = LinearRegression()
            log_model.fit(X_log, values)
            log_pred = log_model.predict(X_log)

            log_r2 = r2_score(values, log_pred)
            log_rmse = np.sqrt(mean_squared_error(values, log_pred))

            y_pred_cv = np.zeros(len(y))            # LOOCV
            for train_idx, test_idx in loo.split(X_log):
                log_model.fit(X_log[train_idx], y[train_idx])
                y_pred_cv[test_idx] = log_model.predict(X_log[test_idx])

            log_rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))

            rss_log = np.sum((y - log_pred)**2)
            aicc_log = compute_aicc(n, rss_log, k=2)

            model_metrics["Log"] = (log_r2, log_rmse, log_rmse_cv, aicc_log)

            print(f" LOG MODEL: coef = {log_model.coef_[0]:.4f}, intercept = {log_model.intercept_:.4f}")
            print(f"    Statistics: R2 = {log_r2:.4f}, RMSE (train) = {log_rmse:.4f}, RMSE (LOOCV) = {log_rmse_cv:.4f}\n")

            # Plot
            x_sorted_idx = np.argsort(X[:,0])
            plt.plot(X[x_sorted_idx], log_pred[x_sorted_idx],
                     linestyle="-", color="#B5D03E", alpha=0.8, label="Log")

        else:
            log_model = None
            log_pred = None

            print(" Log: skipped (X <= 0)")       

        # Finalize combined plot
        #plt.title(f"{phase_name}")
        plt.xlabel("qO₂ [pmol/vc/d]")
        plt.ylabel(var_name)
        plt.legend()
        plt.ylim(0,1.25)
        plt.show()

        # Best model determination
        best_model_cv = min(model_metrics, key=lambda k: model_metrics[k][2])    
        best_model_aicc = min(model_metrics, key=lambda k: model_metrics[k][3]) 

        # Store models and predictions
        for model_name, model_vals  in model_metrics.items():
            summary_rows.append({
                "Phase": phase_name,
                "Variable": var_name,
                "Model": model_name,
                #"Pearson_r": round(r,4),
                #"Pearson_p": round(p,4),
                "R2_train": round(model_vals [0],4),
                "RMSE_train": round(model_vals [1],4),
                "RMSE_LOOCV": round(model_vals [2],4),
                "AICc": round(model_vals[3],4),
                "Best_Model_CV": model_name == best_model_cv,
                "Best_Model_AICc": model_name == best_model_aicc
            })

# Summary table
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(
    by=["Phase", "Variable", "RMSE_LOOCV"]
)#.to_csv("path\\to\\file\\stats_models.csv", sep=";", header=True, decimal=",")

# %%
# EXTRA PLOTS
# ---------------------------------------------------------
#%%
# Plot y_in and y_out
plt.figure(figsize=(10,5))
plt.plot(t_s, y_in_raw*100, label = "O2 in", alpha = 0.8)
plt.plot(t_s, y_out*100, label = "O2 out", alpha = 0.8)
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
plt.ylabel("yO2 [perc-vol]")
plt.legend()
plt.title("")
plt.show()

#%%
# Plot y_in +- sigma
plt.figure(figsize=(8,5))
plt.plot(t_s, y_in_raw*100, label = "O₂ in", alpha = 0.8)
plt.fill_between(t_s, y_in_raw*100 + sigma_yin*100, y_in_raw*100 - sigma_yin*100, label = "σ", alpha = 0.3, color="#24B8D6")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
plt.ylabel("yO2 [perc-vol]")
plt.legend()
plt.title("")
plt.show()

# %%
# Plot SPG-O2 and DO
fig, ax1 = plt.subplots(figsize=(8,5))
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.axvspan(104, 118, color='grey', alpha=0.2, lw=0)
plt.axvspan(122, 142, color='grey', alpha=0.2, lw=0)
plt.axvspan(210, 240, color='grey', alpha=0.2, lw=0)

# Left axis: SPG-O2
ax1.plot(df["t_[h]"], df["SPG-O2_[slpm]"], label=f"SPG O₂", alpha = 0.8)
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("SPG O₂ [slpm]")
ax1.tick_params(axis='y')

# Right axis: DO
ax2 = ax1.twinx()
ax2.plot(df["t_[h]"], df["pO2_[perc-sat]"], label="DO", alpha = 0.8, color = "orange")
ax2.set_ylabel("DO [%-sat]")
ax2.tick_params(axis='y')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.show()

# %%
#Plot: y_in vs y_out
plt.figure(figsize=(10,5))

plt.plot(t_s, y_in_raw, label="yin raw", alpha = 0.5, color = "#FBA904")
plt.plot(t_s, df["y_in_UKF_v2_[perc-vol]"], alpha=0.9, label="yin filt (UKF)", color ="#EE9F0D")

plt.plot(t_s, y_out, label="yout raw", alpha = 0.5, color = "#24B8D6")
plt.plot(t_s, df["y_out_UKF_v2_[perc-vol]"], alpha=0.7, label="yout filt (UKF)", color = "#4985FC")

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlabel("Time [h]")
plt.ylabel("yO2 [%-vol]")
plt.legend()
plt.show()

# %%
# Plot OUR and VCD vs time
fig, ax1 = plt.subplots(figsize=(10,5))

# Left axis: qO₂
#ax1.plot(df["t_[h]"], df["OUR_UKF_v1_[mmol/h/L]"], label=f"OUR (UKF no delay)", color="orange")
ax1.plot(df["t_[h]"], df["OUR_UKF_v2_[mmol/h/L]"], label=f"OUR (UKF delay)")
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("OUR [mmol / h / L]")
ax1.tick_params(axis='y')

# Right axis: VCD
ax2 = ax1.twinx()
ax2.scatter(vcd_df["t_[min]"]/60, vcd_df["VCD_[E6/ml]"], s=1, c="red", label="VCD")
ax2.set_ylabel("VCD [E6/ml]")
ax2.tick_params(axis='y')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlim(0, 325)
plt.title("OUR and VCD over time")
plt.show()

#%%
# Plot calculated qO2_exp vs time
fig, ax1 = plt.subplots(figsize=(8,5))

# Left axis: qO₂
#ax1.plot(t_vcd, qO2_nodelay, label=f"qO₂ (OUR in UKF, no delay)", color = "orange")
ax1.plot(t_vcd, qO2, label=f"qO₂ (OUR in UKF, delay)", color="tab:blue", alpha = 0.8)
#plt.fill_between(t_vcd, qO2 + sigma_qO2, qO2 - sigma_qO2, label = "sigma", alpha = 0.5, color="orange")
#ax1.plot(t_vcd, qO2_exp_post, label=f"qO₂ (OUR pre UKF)", color="red", alpha = 0.8)
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("qO₂ [pmol/vc/d]")
ax1.tick_params(axis='y')
ax1.set_ylim(0,40)

# Right axis: VCD
ax2 = ax1.twinx()
ax2.scatter(vcd_df["t_[min]"]/60, vcd_df["VCD_[E6/ml]"], s=1, c="red", label="VCD")
ax2.set_ylabel("VCD [E6/ml]")
ax2.tick_params(axis='y')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.xlim(0, 325)
plt.title("Single-cell Oxygen Uptake Rate (qO₂) and VCD over time")
plt.show()

#%%
# Plot qO2 smooth
plt.figure(figsize=(5,5))
plt.plot(t_vcd, qO2, color="orange", label="qO₂ raw", alpha=0.5)
plt.plot(t_vcd, qO2_smooth, color="green", label="qO₂ smooth")
plt.ylabel("qO₂ [pmol/vc/d]")
plt.xlabel("Time [h]")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.legend()
plt.show()

# %%
# Plot filtered OUR vs VCD
plt.figure(figsize=(5,5))
plt.scatter(VCD, OUR_min, s=1)
plt.xlabel("VCD [10⁶ cells/mL]")
plt.ylabel("OUR UKF [mmol/h·L]")
plt.title("OUR UKF vs. VCD")
plt.axvline(x=27.5, color='#cd5c5c', linestyle='--')
#plt.xlim(0, 27.5) # For watching ramp-up phase only (0, 27.5) or production only (27.5, end)
#plt.ylim(0,12)
plt.show()

# %%
# Plot calculated qO2 vs VCD
plt.figure(figsize=(5,5))
plt.plot(VCD, qO2, label=f"qO₂ (from OUR UKF)")
#plt.xlim(1.5, 27.5)
#plt.ylim(0, 50)
plt.xlabel("VCD [E6 cell/ml]")
#plt.scatter(vcd_df["t_[min]"]/60, vcd_df["VCD_[E6/ml]"], s=6, c = "red", label="VCD")
plt.ylabel("qO₂ [pmol O₂ / h / cell]")
plt.title("Single-cell Oxygen Uptake Rate (qO₂)")
plt.axvline(x=27.5, color='#cd5c5c', linestyle='--')
plt.show()

# %%
# Choose and plot metabolite of interest
plt.figure(figsize=(8,5))
plt.scatter(metabolites["t_[s]"]/3600, metabolites["HCP_[ng/ml]"])
#plt.plot(metabolites["t_[s]"]/3600, metabolites["qLac_[pmol/vc/d]"])
#plt.ylabel("qS [pmol/c/h]")
plt.xlabel("Concentration [ng/ml]")
plt.xlabel("Time [h]")
plt.title("HCP")
#plt.title("qLactate")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.show()

#%%
# Plot qO2 and Y(O2/Gluc):
fig, ax1 = plt.subplots(figsize=(5,5))

ax1.plot(t_vcd, qO2_smooth, label=f"qO2", color = "green")
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("qO2 [pmol/vc/d]")
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.plot(t_met, Y_O2_Gluc, color = "red", alpha=0.3)
ax2.scatter(t_met, Y_O2_Gluc, label = "Y(O2/Gluc)", color = "red")
ax2.set_ylabel("Yield [-]")
ax2.tick_params(axis='y')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.title("")
plt.xlabel("Time [h]")
plt.show()

#%%
#Plot sanity check: Interpolation to metabolite sample times
plt.figure(figsize=(5,10))

plt.subplot(3,1,1)
plt.title("Interpolation at metabolite sample times")
plt.plot(t_vcd, qO2, label="qO2", alpha=0.6)
plt.scatter(t_met, qO2_met, color="red", label="qO2 interpolated")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.ylabel("qO2 [pmol/vc/d]")
plt.legend()

#Plot sanity check --> VCD
plt.subplot(3,1,2)
plt.plot(vcd_df["t_[h]"], vcd_df["VCD_[E6/ml]"], label="VCD raw", alpha=0.6)
plt.scatter(t_met, VCD_met/1e9, color="red", label="VCD interpolated")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.ylabel("VCD [E6/ml]")
plt.legend()

#Plot sanity check --> D
plt.subplot(3,1,3)
plt.plot(weight["t_[h]"], D, label="D timeline", alpha=0.6)
plt.scatter(t_met, D_met, color="red", label="D interpolated")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.ylabel("D [RV/h]")
plt.legend()

plt.xlabel("Time [h]")
plt.show()

#%%
#Plot: qO2 downsampling check 
plt.plot(t_vcd, qO2, label = "Online", alpha = 0.6)
plt.scatter(t_met, metabolites["qO2_[pmol/vc/d]_avg"], label = "Downsampled")
plt.xlabel("Time [h]")
plt.ylabel("qO2 [pmol/vc/d]")
plt.axvline(t_prod0_h, label='Ramp-up → Production', color='#cd5c5c', linestyle='--')
plt.ylim(0,38)
plt.legend()
