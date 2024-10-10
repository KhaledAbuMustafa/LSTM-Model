import numpy as np
import matplotlib.pyplot as plt
import os


data = np.loadtxt(r"file")

x = data[:, 0]
y = data[:, 1]


def linear_baseline(x):
    baseline = np.zeros_like(x)
    start = 25
    end = 125
    baseline[(x >= start) & (x <= end)] = (x[(x >= start) & (x <= end)] - start) / (end - start) * 7
    baseline[x > end] = baseline[end]
    return baseline

def amplitude_modulation(x):
    amp_mod = np.ones_like(x)
    start = 100
    end = 150

    
    mask = (x >= start) & (x <= end)
    growth = np.linspace(1, 1.5, mask.sum()) 
    amp_mod[mask] = growth

    return amp_mod

baseline_drift = linear_baseline(x)
amp_mod = amplitude_modulation(x)
y_augmented = ((y-np.mean(y)) * amp_mod)+np.mean(y) - baseline_drift

def augment_weight(x):
        #increase and decrease of the augmentation in the range x = 300 to x = 500
    return np.clip((x - 300) / 200, 0, 1) * np.clip((500 - x) / 200, 0, 1)

amplitude_variation = 1 + 0.4 * np.sin(2 * np.pi * 0.005 * x) * augment_weight(x)
#y_augmented = (((y-np.mean(y)) * amp_mod)+np.mean(y) - baseline_drift) * amplitude_variation

plt.plot(x, y_augmented)
plt.xlabel('time[s]')
plt.ylabel('position[mm]')
plt.grid(True)
plt.show()

data = np.column_stack((x, y_augmented))


folder = r"forder_name"   
#os.makedirs(folder, exist_ok=True) 


#file_path = os.path.join(folder, 'augmented_file.txt')


#np.savetxt(file_path, data, fmt='%.6f', delimiter=' ', comments='')