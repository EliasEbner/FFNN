import numpy as np

def main():
    X = np.empty((1000, 1))
    Y = np.empty(X.shape)

    for i in range(X.shape[0]):
        X[i] = i/10
        Y[i] = np.sin(i/10)







if __name__ == '__main__':
    main()
