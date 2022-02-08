import matplotlib.pyplot as plt
import numpy as np

weighted_mean = 0.126
weighted_std = 0.129
pred_z = 0.026
actual_p = 0.026

plt.figure(figsize=(10,6))
#plt.scatter(np.nan, np.nan, label='$\delta$z=0.05 and $\delta$p=0.2') #Largest box for delta_z = 0.05, delta_p = 0.2
#plt.scatter(np.nan, np.nan, label='$\delta$z=0.015 and $\delta$p=0.1') #Medium box for delta_z = 0.015, delta_p = 0.1
plt.errorbar(np.nan, np.nan, marker = 'x', color='r', label='Weighted mean = {0:.3f}\nWeighted std = {1:.3f}\nTarget redshift = {2:.3f}\nActual liklihood = {3:.3f}'.format(weighted_mean, weighted_std, pred_z, actual_p)) #plotting average weighted by 2D gaussian
plt.errorbar(np.nan, np.nan, marker = 'v', color='black', label='Actual Test prediction for new redshift')
plt.errorbar(np.nan, np.nan, marker = 's', color='black', label='Original redshift prediction')



plt.axis('off')
#plt.xticks(alpha=0)
#plt.yticks(alpha=0)
#plt.tick_params(left=False, bottom=False)
plt.legend(fontsize=30, loc='center')

plt.savefig('legend_for_comapring_weighted.png', dpi=200)