import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import beta
import sys

p = 0.2 # Value not in 0.4-0.6
dataset = random.choices([0,1],[p,1-p],k=280) # generate dataset

a,b = 4,6 # Params of Beta
x_lin = np.linspace(0,1,1000)

# Sequential Learning
for index,data in enumerate(dataset):
    if data==0:
        b+=1
    elif data==1:
        a+=1
    else:
        print('Unexpected Data Format')
        sys.exit()
    prob_dist = beta.pdf(x_lin,a,b)

plt.title('Sequential Learning')
plt.plot(x_lin,prob_dist)
plt.savefig('sequential_learning.png')
plt.cla()

# Batch Learning
n_heads, n_tails = 0,0
for data in dataset:
    if data==0:
        n_heads += 1
    elif data==1:
        n_tails += 1
    else:
        print('Unexpected Data Format')
        sys.exit()
prob_dist= beta.pdf(x_lin,a+n_tails,b+n_heads)
plt.title('Batch Learning')
plt.plot(x_lin,prob_dist)
plt.savefig('batch_learning.png')