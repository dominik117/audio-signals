
import numpy as np

np.seterr(invalid='ignore')

def new_model(num_states=10, num_symbols=4):
    """
    Create a new HMM with `num_states` hidden states and
    `num_symbols` observation symbols.  A, B, and pi are
    row-normalized so they’re proper probability distributions.
    """
    pi = np.ones(num_states) / num_states

    A = np.random.rand(num_states, num_states)
    A /= A.sum(axis=1, keepdims=True)

    B = np.random.rand(num_states, num_symbols)
    B /= B.sum(axis=1, keepdims=True)

    return A, B, pi


def forward(O, A, B, pi):
    T, M = len(O), A.shape[0]
    alpha = np.zeros((T, M))
    alpha[0] = pi * B[:, O[0]]
    for t in range(1, T):
        for j in range(M):
            alpha[t, j] = alpha[t-1].dot(A[:, j]) * B[j, O[t]]
    return alpha


def backward(O, A, B):
    T, M = len(O), A.shape[0]
    beta = np.zeros((T, M))
    beta[-1] = 1.0
    for t in range(T-2, -1, -1):
        for i in range(M):
            beta[t, i] = (beta[t+1] * B[:, O[t+1]]).dot(A[i, :])
    return beta


def baum_welch(O, A, B, pi, n_iter=100):
    """
    Re-estimate A, B, and pi by the Baum–Welch (EM) algorithm.
    Returns updated (A, B, pi).
    """
    M, T = A.shape[0], len(O)
    A = A.copy()
    B = B.copy()
    for _ in range(n_iter):
        alpha = forward(O, A, B, pi)
        beta  = backward(O, A, B)

        xi = np.zeros((M, M, T-1))
        for t in range(T-1):
            denom = (alpha[t].reshape(-1,1) * A * B[:, O[t+1]].reshape(1,-1) * beta[t+1]).sum()
            if denom == 0:
                continue
            xi[:,:,t] = (alpha[t].reshape(-1,1) * A * B[:, O[t+1]].reshape(1,-1) * beta[t+1]) / denom

        gamma = xi.sum(axis=1)               # shape (M, T-1)
        gamma = np.hstack([gamma, gamma[:,-1].reshape(-1,1)])  # add last column

        # update pi
        pi = gamma[:,0] / gamma[:,0].sum()

        # update A
        A = xi.sum(axis=2) / gamma[:,:-1].sum(axis=1).reshape(-1,1)
        A[np.isnan(A)] = 0.0

        # update B
        for k in range(B.shape[1]):
            B[:,k] = gamma[:, O==k].sum(axis=1)
        B = np.divide(B, gamma.sum(axis=1).reshape(-1,1))
        B[np.isnan(B)] = 0.0

    return A, B, pi


def viterbi(O, A, B, pi):
    T, M = len(O), A.shape[0]
    delta = np.zeros((T, M))
    psi   = np.zeros((T-1, M), dtype=int)

    delta[0] = pi * B[:, O[0]]
    for t in range(1, T):
        for j in range(M):
            probs = delta[t-1] * A[:, j] * B[j, O[t]]
            psi[t-1, j] = np.argmax(probs)
            delta[t, j] = np.max(probs)

    # backtrace
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        states[t] = psi[t, states[t+1]]

    return states
