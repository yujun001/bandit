import math
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import inv
from tqdm import tqdm

def read_data(filename):
    '''
    Input: path to csv file
    Output: numpy arrays of action, reward and context
    Description: Read csv to dataframe. Parses the dataframe 
                 and converts to numpy arrays.
    '''
    df = pd.read_csv(filename, header=None, sep="\s+")
    action = df.iloc[:,0].values
    reward = df.iloc[:,1].values
    x_ta = df.iloc[:,2:].values

    return action, reward, x_ta

def initialize_params(d, num_actions):
    '''
    Input: time-steps and number of dimensions
    Output: A_a and b_a
    Description: Initialize A_a to identity matrix for each action.
                 Initialize b_a to 0 d-dimensional vector
    '''
    A = [np.identity(d) for i in range(num_actions)]
    b_as = [np.zeros(d) for i in range(num_actions)]
    print(A)
    print(b_as)

    return A, b_as

def linUCB(action, reward, x_ta, A, b_as, alphas):
    ''''
    Input:  action, reward and context features for each timestep in the dataset.
            Initialized values for A and b_as.
            Alphas are each timestep for a particular alpha strategy.
    Output: C(T) as an array of size Tx1, where T is the number of 
            time-steps in the dataset.
    Description: perform a LinUCB with disjoint linear models which is 
                modified to work in an online manner on the dataset given. 
                And compute C(t) at each timestep t.
    '''

    c_t_num = []
    c_t_den = []

    for i in tqdm(range(time_steps)):
        alpha = alphas[i]
        x_ta_i = x_ta[i,:]
        r_i = reward[i]
        action_i = action[i]

        p_t_a = [0.] * 10
        for a_i in range(10):
            inv_a = inv(A[a_i])
            theta = np.dot(inv_a, b_as[a_i])
            p_t_a[a_i] = np.dot(theta.T, x_ta_i) + alpha * \
                        sqrt(np.dot(np.dot(x_ta_i.T, inv_a), x_ta_i))

        # choose the best action
        a_t = np.argmax(p_t_a)

        if i >= 1:
            # update
            if a_t + 1 == action_i:

                x_ta_i = x_ta_i.reshape(100, 1)
                A[a_t] +=  np.dot(x_ta_i, x_ta_i.T)
                b_as[a_t] +=  r_i * x_ta_i.flatten()

                c_t_num.append(r_i)
                c_t_den.append(1.)
            else:
                c_t_num.append(0.)
                c_t_den.append(0.)

    c_t_num = np.cumsum(c_t_num)
    c_t_den = np.cumsum(c_t_den)

    np.seterr(all = 'ignore')
    c_t = np.nan_to_num(np.divide(c_t_num, c_t_den, dtype = float))

    return c_t
 
def plot_ct(c_t_1_t, c_t_sqrt, c_01sqrt, c_e_t, c_point01):
    '''
    Input: C(T) for alpha strategies
    Description: Plot the C(T) for each alpha strategy as time-steps progress.
                The resulting plot is saved on disk
    '''
    p1, = plt.plot(range(1, len(c_t_1_t) + 1), c_t_1_t, 'r', 
        label = "alpha = 1/t")
    p2, = plt.plot(range(1, len(c_t_sqrt) + 1), c_t_sqrt, 'g', 
        label = "alpha = 1/sqrt(t)")
    p3, = plt.plot(range(1, len(c_01sqrt) + 1), c_01sqrt, 'y', 
        label = "alpha = .1/sqrt(t)")
    p4, = plt.plot(range(1, len(c_e_t) + 1), c_e_t, 'b', 
        label = "alpha = e^(-t)")
    p5, = plt.plot(range(1, len(c_point01)+1), c_point01, 'k', 
        label = "alpha = 0.1")
    
    plt.legend(handles = [p1, p2, p3, p4, p5])

    plt.xlabel("Timestep (T)", fontsize = 14)
    plt.ylabel("Cumulative take-rate replay C(T)", fontsize = 14)
    plt.tight_layout()
    plt.savefig("plots_new", dpi = 500)
    plt.close()


if __name__ =='__main__':

    # read the dataset
    action, reward, x_ta = read_data("dataset.txt")

    # determine dataset specific values
    d = x_ta.shape[1]
    time_steps = action.shape[0]
    num_actions = 10
    print(d,time_steps)

    # use different alpha strategies
    alphas_1_t = [1./(i + 1.0) for i in range(time_steps)]

    alphas_sqrt = [1./sqrt(i + 1.0) for i in range(time_steps)]
    alphas_01sqrt = [0.1/sqrt(i + 1.0) for i in range(time_steps)]
    alphas_e_t = [math.exp(-i) for i in range(time_steps)]
    alphas_point01 = [0.1 for i in range(time_steps)]

    print(alphas_1_t)

    # run linUCB for different alphas
    A_init , b_as_init = initialize_params(d,num_actions)
    c_t_1_t = linUCB(action, reward, x_ta, A_init, b_as_init, alphas_1_t)
    print("Final C(T) %s %f " %("alpha = 1/t", c_t_1_t[-1]))

    # A_init , b_as_init = initialize_params(d,num_actions)
    # c_t_sqrt = linUCB(action, reward, x_ta, A_init, b_as_init, alphas_sqrt)
    # print("Final C(T) %s %f " %("alpha = 1/sqrt(t)", c_t_sqrt[-1]))
    #
    # A_init , b_as_init = initialize_params(d,num_actions)
    # c_e_t = linUCB(action, reward, x_ta, A_init , b_as_init, alphas_e_t)
    # print("Final C(T) %s %f " %("alpha = e^(-t)", c_e_t[-1]))
    #
    # A_init , b_as_init = initialize_params(d,num_actions)
    # c_point01 = linUCB(action, reward, x_ta, A_init, b_as_init, alphas_point01)
    # print("Final C(T) %s %f " %("alpha = 0.1", c_point01[-1]))
    #
    # A_init , b_as_init = initialize_params(d,num_actions)
    # c_01sqrt = linUCB(action, reward, x_ta, A_init, b_as_init, alphas_01sqrt)
    # print("Final C(T) %s %f " %("alpha = .1/sqrt(t)", c_01sqrt[-1]))
    #
    # # plot the result for C(T)
    # plot_ct(c_t_1_t, c_t_sqrt, c_01sqrt, c_e_t, c_point01)
