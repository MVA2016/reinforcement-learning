# Import Packages
import numpy as np

M = 15  # size of the stock
K = 0.8  # delivery cost
h = 0.3  # maintenance cost
c = 0.5  # buying price
p = 1  # selling price
q = 0.1  # parameter of the geometric distribution
gamma = 0.98  # discount factor (inflation rate)
pi1 = 2 * np.ones(M + 1).astype(int)  # constant policy
# pi=2*np.ones(M)


def geometric_pmf(D=15, q=q):
    """ Return a a tuple of ndarray with the k,P(X=k) associate with a truncated geometric law in N
    Used for the distribution of customer """
    labels = np.arange(
        1, D + 2)  # labels goes from 0 to D +1 , this is not a geometric law
    # geometric law is not defined in 0
    probs = np.array([q * (1 - q) ** (i - 1) for i in labels])
    probs[-1] = 1 - probs[:D].sum()  # truncate the distribution in label D
    return [l - 1 for l in labels], probs


def simu(pmf=geometric_pmf()):
    """ Draw one sample from of a discrete distribution, pmf is supposed to
    be in ascending order  """
    labels = pmf[0]
    probs = pmf[1]
    u = np.random.rand()
    return labels[int((u >= probs.cumsum()).argmin())]


# def simu2(q=q,nb_sample=1):
#     """ Draw nb_sample sample from the customer distribution """
#     sample = np.random.geometric(q,nb_sample)
#     sample[sample >= M] = M
#     return sample


def next_state(x, a, d, m=M):
    """ Return next state """
    return max(0, min((x + a), M) - d)


def next_reward(x, a, d, m=M, K=K, h=h, c=c, p=p):
    """ Return next reward """
    return -K * (a > 0) - c * max(0, min(x + a, M) - x) - h * x + p * min(d, x + a, M)


def simu_transition(x, a, M=M, K=K, h=h, c=c, p=p):
    """ Simulate a transition, return a tuple(new_state,reward) """
    d = simu()  # simulate the number of customers that arrive this week
    return next_state(x, a, d=d, m=M), next_reward(x, a, d=d, m=M, K=K, h=h, c=c, p=p)


def mdp(D=geometric_pmf()[1], M=M, K=K, h=h, c=c, p=p):
    """ Return transition model of and reward of MDP

     Parameters
     ----------
     D: ndarray
        probability vector
     M,h,c,K : integers
        parameters from the model

     Returns
     -------
     tuple
        P[x,y,a] et R[x,a] transition probability matrix and reward matrix
     """
    P = np.zeros([M + 1, M + 1, M + 1])
    R = np.zeros([M + 1, M + 1])

    for a in range(M + 1):
        for x in range(M + 1):
            for d in range(M + 1):  # go trough every value of D

                P[x, next_state(x, a, d), a] += D[d]
                R[x, a] += D[d] * next_reward(x, a, d)
    return P, R

# compute mdp
P,R = mdp()
