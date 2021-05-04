import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here


    dividends = 200
    x = np.linspace(-1, 1, dividends)
    y = np.linspace(-1, 1, dividends)
    X, Y = np.meshgrid(x, y)

    samples = []
    for i in y:
        for j in x:
            samples.append([j, i])
    samples = np.asarray(samples)
    density = util.density_Gaussian([0, 0], [[beta, 0], [0, beta]], samples)
    Z = density.reshape(X.shape)

    # # plot the contours
    plt.contour(X, Y, Z, colors='b')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.plot([-0.1], [-0.5], marker='o', markersize=6, color='orange')
    plt.title('prior distribution')
    plt.savefig("prior.pdf")
    plt.show()
    return


def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here

    cov_a_inv = [[1 / beta, 0], [0, 1 / beta]]
    A = np.append(np.ones(shape=(len(x), 1)), x, axis=1)
    print(A.shape)

    cov_abyz = np.linalg.inv(cov_a_inv + np.dot(A.T, A) / sigma2)
    mu_abyz = np.dot(cov_abyz, np.dot(A.T, z)/sigma2).squeeze()

    mu = mu_abyz
    Cov = cov_abyz

    dividends = 200
    x1 = np.linspace(-1, 1, dividends)
    y1 = np.linspace(-1, 1, dividends)
    X, Y = np.meshgrid(x1, y1)

    samples = []
    for i in y1:
        for j in x1:
            samples.append([j, i])
    samples = np.asarray(samples)
    density = util.density_Gaussian(mu.T, Cov, samples)
    Z = density.reshape(X.shape)

    plt.contour(X, Y, Z, colors='b')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title("posterior distribution based on " + str(x.shape[0]) +" samples")
    plt.plot([-0.1], [-0.5], marker='o', markersize=6, color='orange')
    plt.savefig("posterior" + str(x.shape[0]) + ".pdf")
    plt.show()

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here

    A = np.append(np.ones(shape=(len(x), 1)), np.expand_dims(x, 1), axis=1)

    mu_z = np.dot(A, mu)
    cov_z = sigma2 + np.dot(A, np.dot(Cov, A.T))
    std_z = np.sqrt(np.diag(cov_z))

    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.scatter(x_train, z_train, color='blue')
    plt.errorbar(x, mu_z, yerr=std_z, fmt='ro')

    plt.title("prediction using " + str(x_train.shape[0]) + " samples")
    plt.savefig("predict1.pdf")

    plt.show()
    return

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 1
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    
