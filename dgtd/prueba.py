import numpy as np
import scipy.special
import math

k_elem = 4
lst = [np.ones(int(k_elem/2)), 2*np.ones(int(k_elem/2))]

print (lst)

# r = np.array([-1. , -0.4472136 , 0.4472136 , 1.0])
# alpha = 0
# beta = 0 
# N = 3


# PL = np.zeros([N+1,len(r)]) 
# # Initial values P_0(x) and P_1(x)
# gamma0 = 2**(alpha+beta+1) \
#             / (alpha+beta+1) \
#             * scipy.special.gamma(alpha+1) \
#             * scipy.special.gamma(beta+1) \
#             / scipy.special.gamma(alpha+beta+1);
# PL[0] = 1.0 / math.sqrt(gamma0);
# if N == 0:
#     print(PL.transpose)

# gamma1 = (alpha+1.) * (beta+1.) / (alpha+beta+3.) * gamma0;
# PL[1] = ((alpha+beta+2.)*r/2. + (alpha-beta)/2.) / math.sqrt(gamma1);

# if N == 1:
#     print(PL.transpose)

# # Repeat value in recurrence.
# aold = 2. / (2.+alpha+beta) \
#         * math.sqrt( (alpha+1.)*(beta+1.) / (alpha+beta+3.));

# # Forward recurrence using the symmetry of the recurrence.
# for i in range(N-1):
#     h1 = 2.*(i+1.) + alpha + beta;
#     anew = 2. / (h1+2.) \
#             * math.sqrt((i+2.)*(i+2.+ alpha+beta)*(i+2.+alpha)*(i+2.+beta) \
#                         / (h1+1.)/(h1+3.));
#     bnew = - (alpha**2 - beta**2) / h1 / (h1+2.);
#     PL[i+2] = 1. / anew * (-aold * PL[i] + (r-bnew) * PL[i+1]);
#     aold = anew;

# print(PL)