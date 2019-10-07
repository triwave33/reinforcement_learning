import numpy as np
import matplotlib.pyplot as plt
import glob

# feature num
fn_files = glob.glob('feature_num_test/reward_list*.npy*')

numbers = [int(i.split('N_')[-1].split('.npy')[0]) for i in fn_files]

reward_list = [np.load(i) for i in fn_files]
average_list = [np.mean(i) for i in reward_list]

for n,a in zip(numbers, average_list):
    plt.scatter(n,a, color='k', marker='x')

plt.title('effect of feature num')
plt.xlabel('# of features')
plt.savefig('feature.png')
plt.close()

#for i in [0,5,7]:
#    plt.plot(reward_list[i], label=str(numbers[i]), alpha=1)
#    plt.legend(loc='best')
#plt.show()


# sigma num
fn_files = glob.glob('sigma_test/reward_list*.npy*')
numbers = [int(i.split('s_0')[-1].split('.npy')[0])/1000. for i in fn_files]

reward_list = [np.load(i) for i in fn_files]
average_list = [np.mean(i) for i in reward_list]

for n,a in zip(numbers, average_list):
    plt.scatter(n,a, color='k', marker='x')
plt.title('effect of sigma')
plt.xlabel('sigma')
plt.ylabel('average reward')
plt.savefig('sigma.png')
plt.close()

# gamma
fn_files = glob.glob('gamma_test/reward_list*.npy*')
numbers = [int(i.split('g_')[-1].split('.npy')[0])/1000. for i in fn_files]

reward_list = [np.load(i) for i in fn_files]
average_list = [np.mean(i) for i in reward_list]

for n,a in zip(numbers, average_list):
    plt.scatter(n,a, color='k', marker='x')
plt.title('effect of gamma')
plt.xlabel('gamma')
plt.ylabel('average reward')
plt.savefig('gamma.png')
plt.close()

# lambda
fn_files = glob.glob('lambda_test/reward_list*.npy*')
numbers = [int(i.split('l_')[-1].split('.npy')[0])/100. for i in fn_files]

reward_list = [np.load(i) for i in fn_files]
average_list = [np.mean(i) for i in reward_list]

for n,a in zip(numbers, average_list):
    plt.scatter(n,a, color='k', marker='x')
plt.title('effect of lambda')
plt.xlabel('lambda')
plt.ylabel('average reward')
plt.savefig('lambda.png')
plt.close()


# alpha
fn_files = glob.glob('alpha_test/reward_list*.npy*')
numbers = [int(i.split('a_')[-1].split('.npy')[0])/10000. for i in fn_files]

reward_list = [np.load(i) for i in fn_files]
average_list = [np.mean(i) for i in reward_list]

for n,a in zip(numbers, average_list):
    plt.scatter(n,a, color='k', marker='x')
plt.title('effect of alpha')
plt.xlabel('alpha')
plt.ylabel('average reward')
plt.savefig('alpha.png')
plt.close()


# epsilon
fn_files = glob.glob('epsilon_test/reward_list*.npy*')
numbers = [float(i.split('list_')[-1].split('.npy')[0]) for i in fn_files]

reward_list = [np.load(i) for i in fn_files]
average_list = [np.mean(i) for i in reward_list]
average_list2 = [np.mean(i[-200:]) for i in reward_list]

for n,a in zip(numbers, average_list2):
    plt.scatter(n,a, color='k', marker='x')
plt.title('effect of epsilon')
plt.xlabel('epsilon')
plt.ylabel('average reward')
plt.savefig('epsilon.png')
plt.close()

#for n,r in zip(numbers, reward_list):
#    plt.plot(r, label=str(n))
#plt.show()
#plt.title('effect of epsilon')
#plt.xlabel('epsilon')
#plt.ylabel('average reward')

