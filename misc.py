
#%%
# 
import numpy as np
import matplotlib.pyplot as plt


sigma = lambda x: 1/(1+np.exp(-x))

# approx = lambda M, a, b: sigma(a / np.sqrt(1 + np.pi**2*b**2/3))**M
# approx = lambda M, a, b: sigma(a / np.sqrt(1 + np.pi*b**2/8))**M
approx = lambda M, a, b: sigma(2.0 * a / np.sqrt(1 + np.pi*b**2/8))**M

Ms = np.arange(0, 21)
K = 100000

a = 0.3
b = 0.5

x = np.random.normal(a,b, K)

exp_approx = approx(Ms, a, b)
exp_emp = (sigma(x)[:, None]**Ms[None, :]).mean(0)


plt.figure()
plt.plot(Ms, exp_approx, label = "approx")
plt.plot(Ms, exp_emp, label = "empirical")
plt.legend()
plt.show()
plt.close()



# %%
