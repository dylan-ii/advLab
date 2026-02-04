import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# table I: distance d measurements (cm)
d_measurements = np.array([11.38, 11.17, 11.32, 11.40, 11.23, 11.18, 11.11, 11.22])

# table II: diameter h measurements (cm)
h_measurements = np.array([4.02, 4.01, 4.01, 4.01, 4.01, 4.01, 4.01, 4.01, 4.01, 4.01])

# table III: a+t and t measurements (cm)
a_plus_t = np.array([3.89, 3.90, 3.94, 3.92, 3.95, 3.95, 3.92, 3.93, 3.93, 3.95])
t_measurements = np.array([0.62, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63])

# table IV: temperature vs voltage
temp_voltage_data = {
    'Temperature_C': [22.2, 70, 130, 181, 253, 315, 362, 408],
    'Voltage_mV': [0.060, 0.279, 0.792, 1.355, 2.473, 3.955, 5.457, 7.223]
}

# table V: angle vs voltage 
angle_voltage_data = {
    'Angle_deg': [-10, 0, 10, 20, 30, 40, 50, 60],
    'Temperature_C': [406, 408, 399, 398, 400, 403, 404, 404],
    'Voltage_mV': [6.540, 7.223, 6.775, 6.499, 5.883, 5.173, 4.181, 3.198]
}

APERTURE_DIAMETER = 2.0  # cm (from water-cooled shield)
SENSOR_DIAMETER = 1.5    # cm (15 mm diameter surface)
THERMOPILE_RESPONSIVITY = 0.28  # V/W (value from guide)

d_mean = np.mean(d_measurements)
d_std = np.std(d_measurements, ddof=1)
d_error = d_std / np.sqrt(len(d_measurements))

h_mean = np.mean(h_measurements)
h_std = np.std(h_measurements, ddof=1)
h_error = h_std / np.sqrt(len(h_measurements))

a_t_mean = np.mean(a_plus_t)
a_t_std = np.std(a_plus_t, ddof=1)
a_t_error = a_t_std / np.sqrt(len(a_plus_t))

t_mean = np.mean(t_measurements)
t_std = np.std(t_measurements, ddof=1)
t_error = t_std / np.sqrt(len(t_measurements))

a_mean = a_t_mean - t_mean
a_error = np.sqrt(a_t_error**2 + t_error**2)

r_mean = d_mean - h_mean/2
r_error = np.sqrt(d_error**2 + (h_error/2)**2)

R_mean = r_mean + a_mean
R_error = np.sqrt(r_error**2 + a_error**2)

print(f"\nparams:")
print(f"d (oven aperture to thermopile edge): {d_mean:.3f} ± {d_error:.3f} cm")
print(f"h (thermopile diameter): {h_mean:.3f} ± {h_error:.3f} cm")
print(f"a (opening to sensor): {a_mean:.3f} ± {a_error:.3f} cm")
print(f"r (aperture to opening center): {r_mean:.3f} ± {r_error:.3f} cm")
print(f"R (total distance aperture to sensor): {R_mean:.3f} ± {R_error:.3f} cm")

background_temp_C = 22.2
background_temp_K = background_temp_C + 273.15
background_voltage_mV = 0.060

temps_C = np.array(temp_voltage_data['Temperature_C'])
temps_K = temps_C + 273.15
voltages_full = np.array(temp_voltage_data['Voltage_mV']) 

T0 = background_temp_K
V0 = background_voltage_mV

T4_minus_T0_4 = temps_K**4 - T0**4

V_minus_V0 = voltages_full - V0

A = np.sum(V_minus_V0 * T4_minus_T0_4) / np.sum(T4_minus_T0_4**2)

SS_res = np.sum((V_minus_V0 - A*T4_minus_T0_4)**2)
SS_tot = np.sum((V_minus_V0 - np.mean(V_minus_V0))**2)
R2 = 1 - SS_res/SS_tot

log_T4_diff = np.log(T4_minus_T0_4[1:])
log_V_diff = np.log(V_minus_V0[1:])

slope_log, intercept_log, r_log, _, _ = linregress(log_T4_diff, log_V_diff)

#angle analysis
voltages_angle_full = np.array(angle_voltage_data['Voltage_mV'])
V0_angle = background_voltage_mV
V_angle_minus_V0 = voltages_angle_full - V0_angle

angles = np.array(angle_voltage_data['Angle_deg'])
angles_rad = np.radians(angles)
cos_angles = np.cos(angles_rad)

slope_angle, intercept_angle, r_angle, p_value, std_err = linregress(cos_angles, V_angle_minus_V0)

# stefan-boltzmann constant estimation
R_m = R_mean / 100  # cm -> m
aperture_radius = APERTURE_DIAMETER / 2 / 100  # cm -> m
sensor_radius = SENSOR_DIAMETER / 2 / 100  # cm -> m

aperture_area_m2 = np.pi * (aperture_radius)**2
sensor_area_m2 = np.pi * (sensor_radius)**2

slope_V = A / 1000  # mV -> V

# from guide: σ = (slope * piR^2) / (resp * A_ap * A_det)
sigma_est = (slope_V * np.pi * R_m**2) / (THERMOPILE_RESPONSIVITY * aperture_area_m2 * sensor_area_m2)

print(f"\nresults:")
print(f"  estimated stefan-boltzmann constant: {sigma_est:.2e} W/m²K⁴")
print(f"  expected value: 5.67e-08 W/m²K⁴")
print(f"  ratio: {sigma_est/5.67e-08:.2f}")

print(f"  exponent for T^4 dependence (multiply by 4): {slope_log*4:.3f}")

# plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# panel A: angle vs voltage
ax1 = axes[0, 0]
ax1.scatter(angles, V_angle_minus_V0, color='blue', s=50)
angles_smooth = np.linspace(-10, 60, 100)
cos_fit = slope_angle * np.cos(np.radians(angles_smooth)) + intercept_angle
ax1.plot(angles_smooth, cos_fit, 'r--', label=f'Fit: V = {slope_angle:.1f}cosθ + {intercept_angle:.2f}\nR² = {r_angle**2:.3f}')
ax1.set_xlabel('Angle θ (degrees)', fontweight='bold', fontsize=20)
ax1.set_ylabel('Voltage (mV)', fontweight='bold', fontsize=20)
ax1.set_title('Cosine Dependence', fontweight='bold', fontsize=30)
ax1.legend(fontsize=15)
ax1.grid(True, alpha=0.3)

# panel B: voltage vs cos(angle)
ax2 = axes[0, 1]
ax2.scatter(cos_angles, V_angle_minus_V0, color='green', s=50)
cos_range = np.linspace(min(cos_angles), max(cos_angles), 100)
ax2.plot(cos_range, slope_angle * cos_range + intercept_angle, 'r--', 
         label=f'Linear fit: V = {slope_angle:.1f}cosθ + {intercept_angle:.2f}\nR² = {r_angle**2:.3f}')
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.set_xlabel('cos(θ)', fontweight='bold', fontsize=20)
ax2.set_ylabel('Voltage (mV)', fontweight='bold', fontsize=20)
ax2.set_title('V ∝ cos(θ)', fontweight='bold', fontsize=30)
ax2.legend(fontsize=15)
ax2.grid(True, alpha=0.3)

# panel C: temperature vs voltage (corrected)
ax3 = axes[1, 0]
ax3.scatter(temps_C, V_minus_V0, color='red', s=50, alpha=0.5)
temp_fit = np.linspace(min(temps_C), max(temps_C), 100)
temp_fit_K = temp_fit + 273.15
T4_diff_fit = temp_fit_K**4 - T0**4
voltage_fit = A * T4_diff_fit
ax3.plot(temp_fit, voltage_fit, 'b--', 
         label=f'Fit: V = {A:.2e}×(T⁴-T₀⁴)\nR² = {R2:.3f}')
ax3.set_xlabel('Temperature (°C)',fontweight='bold', fontsize=20)
ax3.set_ylabel('Voltage (mV)', fontweight='bold', fontsize=20)
ax3.set_title('Stefan-Boltzmann Law', fontweight='bold', fontsize=30)
ax3.legend(fontsize=15)
ax3.grid(True, alpha=0.3)

# panel D: V-V0 vs (T⁴-T0⁴)
ax4 = axes[1, 1]
ax4.scatter(T4_minus_T0_4, V_minus_V0, color='purple', s=50)
T4_range = np.linspace(min(T4_minus_T0_4), max(T4_minus_T0_4), 100)
ax4.plot(T4_range, A * T4_range, 'g--',
         label=f'Linear fit: V = {A:.2e}×(T⁴-T₀⁴)\nR² = {R2:.6f}\nSlope = {A:.2e} mV/K⁴')
ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax4.set_xlabel('T⁴ (K⁴)', fontweight='bold',fontsize=20)
ax4.set_ylabel('Voltage (mV)', fontweight='bold', fontsize=20)
ax4.set_title('V ∝ (T⁴)', fontweight='bold', fontsize=30)
ax4.legend(fontsize=15)
ax4.grid(True, alpha=0.3)

plt.suptitle('Black Body Radiation Experiment Analysis', fontsize=40, fontweight='bold')
plt.tight_layout()
plt.show()