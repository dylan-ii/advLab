import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PLOT_FONT_FAMILY = 'Arial'                
PLOT_TITLE_FONTSIZE = 50                 
PLOT_LABEL_FONTSIZE = 40    
PLOT_TICK_FONTSIZE = 35      
PLOT_LEGEND_FONTSIZE = 35 

plt.rcParams['font.sans-serif'] = [PLOT_FONT_FAMILY]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = PLOT_TICK_FONTSIZE
plt.rcParams['axes.labelsize'] = PLOT_LABEL_FONTSIZE
plt.rcParams['axes.titlesize'] = PLOT_TITLE_FONTSIZE
plt.rcParams['legend.fontsize'] = PLOT_LEGEND_FONTSIZE
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = PLOT_TICK_FONTSIZE
plt.rcParams['ytick.labelsize'] = PLOT_TICK_FONTSIZE

csv_file = "measurements.csv"

# consts for e/m calculation
R = 0.158 #m
N = 130
mu0 = 4 * np.pi * 1e-7
factor_numerator = 2 * ((5/4)**3) * R**2   # 2 * ( (5/4)^3 * R^2 )

sigma_U = 1.0 # V
sigma_I = 0.01 # A
sigma_r_cm = 0.1 # cm
sigma_r_m = sigma_r_cm / 100.0

df = pd.read_csv(csv_file)

df['radius_cm'] = df['d'] / 2.0

df['radius_m'] = df['radius_cm'] / 100.0

# e/m = factor_numerator * U / (N * mu0 * I * r)^2
df['e_over_m'] = ( factor_numerator * df['U'] ) / ( (N * mu0 * df['I'] * df['radius_m'])**2 )

# (sigma_em/em)^2 = (sigma_U/U)^2 + (2*sigma_I/I)^2 + (2*sigma_r/r)^2
frac_sq = (sigma_U / df['U'])**2 + (2 * sigma_I / df['I'])**2 + (2 * sigma_r_m / df['radius_m'])**2
df['e_over_m_err'] = df['e_over_m'] * np.sqrt(frac_sq)

print("\n" + "="*70)
print("e/m (C/kg)")
print("="*70)

voltages = sorted(df['U'].unique())

for U in voltages:
    subset = df[df['U'] == U]
    mean_em = subset['e_over_m'].mean()

    sem_em = np.sqrt((subset['e_over_m_err']**2).sum()) / len(subset)

    std_em = subset['e_over_m'].std()
    count = len(subset)

    print(f"Voltage {U} V:")
    print(f"  mean e/m = {mean_em:.3e}  ± {sem_em:.3e}")
   
overall_mean = df['e_over_m'].mean()
overall_sem = np.sqrt((df['e_over_m_err']**2).sum()) / len(df)
overall_std = df['e_over_m'].std()
print("-" * 70)
print(f"Overall:")
print(f"  mean e/m = {overall_mean:.3e}  ± {overall_sem:.3e}")
print("="*70 + "\n")

plt.figure(figsize=(10, 6))

colors = plt.cm.viridis(np.linspace(0, 1, len(voltages)))

regression_lines = []

for i, voltage in enumerate(voltages):

    subset = df[df['U'] == voltage]
    x = subset['radius_cm'].values  #radius in cm for plotting
    y = subset['I'].values

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)

        print(f"Voltage {voltage} V: slope = {slope:.6e}, intercept = {intercept:.6e}")

        x_fit = np.linspace(x.min(), x.max(), 50)
        y_fit = slope * x_fit + intercept

        plt.plot(x_fit, y_fit, color=colors[i], linestyle='-',
                 label=f'U = {voltage} V (fit)')
        regression_lines.append((voltage, slope, intercept))
    else:
        print(f"Warning: Not enough points for voltage {voltage} to fit a line.")

    plt.scatter(x, y, color=colors[i], marker='o',
                label=f'U = {voltage} V (data)')

plt.xlabel('Radius (cm)')
plt.ylabel('Current (A)')
plt.title('Current vs. Radius for Different Voltages')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()