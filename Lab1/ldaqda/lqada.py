import numpy as np
import matplotlib.pyplot as plt
import util


def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models

    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples

    y: a N-by-1 1D array contains the labels of the N samples

    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA
    and 1 figure for QDA in this function
    """
    ### TODO: Write your code here

    # initialization
    mu_male_height = 0
    mu_male_weight = 0
    mu_female_height = 0
    mu_female_weight = 0
    num_male = 0
    num_female = 0
    male_height = []
    male_weight = []
    female_height = []
    female_weight = []

    # get number of male and female and sum of height and weight for male and female, and list of information
    for item in range(0, len(y)):
        if y[item] == 1:
            num_male = num_male + 1
            mu_male_height = mu_male_height + x[item][0]
            male_height.append(x[item][0])
            mu_male_weight = mu_male_weight + x[item][1]
            male_weight.append(x[item][1])
        else:
            num_female = num_female + 1
            mu_female_height = mu_female_height + x[item][0]
            female_height.append(x[item][0])
            mu_female_weight = mu_female_weight + x[item][1]
            female_weight.append(x[item][1])

    # calculate mu_male, mu_female, and prepare mean for all people
    mu_male = [mu_male_height / num_male, mu_male_weight / num_male]
    mu_female = [mu_female_height / num_female, mu_female_weight / num_female]
    num_total = len(y)
    mu_height = (mu_male_height + mu_female_height) / num_total
    mu_weight = (mu_male_weight + mu_female_weight) / num_total

    # initialization
    cov_0_0 = 0
    cov_0_1 = 0
    cov_1_1 = 0

    # calculate cov matrix for both genders
    for item in range(0, len(y)):
        cov_0_0 = cov_0_0 + (x[item][0] - mu_height) ** 2
        cov_0_1 = cov_0_1 + (x[item][0] - mu_height) * (x[item][1] - mu_weight)
        cov_1_1 = cov_1_1 + (x[item][1] - mu_weight) ** 2

    cov = [[cov_0_0 / num_total, cov_0_1 / num_total], [cov_0_1 / num_total, cov_1_1 / num_total]]

    # initialization
    cov_male_0_0 = 0
    cov_male_0_1 = 0
    cov_male_1_1 = 0
    cov_female_0_0 = 0
    cov_female_0_1 = 0
    cov_female_1_1 = 0

    # calculate cov matrix for male and female
    for item in range(0, len(y)):
        if y[item] == 1:
            cov_male_0_0 = cov_male_0_0 + (x[item][0] - mu_male[0]) ** 2
            cov_male_0_1 = cov_male_0_1 + (x[item][0] - mu_male[0]) * (x[item][1] - mu_male[1])
            cov_male_1_1 = cov_male_1_1 + (x[item][1] - mu_male[1]) ** 2
        else:
            cov_female_0_0 = cov_female_0_0 + (x[item][0] - mu_female[0]) ** 2
            cov_female_0_1 = cov_female_0_1 + (x[item][0] - mu_female[0]) * (x[item][1] - mu_female[1])
            cov_female_1_1 = cov_female_1_1 + (x[item][1] - mu_female[1]) ** 2

    cov_male = [[cov_male_0_0 / num_male, cov_male_0_1 / num_male], [cov_male_0_1 / num_male, cov_male_1_1 / num_male]]
    cov_female = [[cov_female_0_0 / num_female, cov_female_0_1 / num_female],
                  [cov_female_0_1 / num_female, cov_female_1_1 / num_female]]

    print(mu_male)
    print(mu_female)
    print(cov)
    print(cov_male)
    print(cov_female)

    # plot part for lda
    # plot all datapoints
    set_xlim = [50, 80]
    set_ylim = [80, 280]
    plt.scatter(male_height, male_weight, color='blue')
    plt.scatter(female_height, female_weight, color='red')
    print(male_height, male_weight)
    # plot gradient
    x_grid = np.linspace(50, 80, 100)
    y_grid = np.linspace(80, 280, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    male_lda = []
    male_qda = []
    female_lda = []
    female_qda = []
    x_set = []
    xx = X[0].reshape(100, 1)
    # make data points samples to pass into density_Gaussian
    for i in range(0, 100):
        samples = np.concatenate((xx, Y[i].reshape(100, 1)), 1)
        male_lda.append(util.density_Gaussian(mu_male, cov, samples))
        female_lda.append(util.density_Gaussian(mu_female, cov, samples))
        male_qda.append(util.density_Gaussian(mu_male, cov_male, samples))
        female_qda.append(util.density_Gaussian(mu_female, cov_female, samples))

    # plot the contours
    plt.contour(X, Y, male_lda, colors='b')
    plt.contour(X, Y, female_lda, colors='r')

    # plot the decision boundary
    boundary_lda = np.asarray(male_lda) - np.asarray(female_lda)
    plt.contour(X, Y, boundary_lda, 0, color='k')
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('contour and decision boundary for lda')
    plt.savefig("lda.pdf")
    plt.show()

    # plot part for qda
    # plot all data points
    set_xlim = [50, 80]
    set_ylim = [80, 280]
    plt.scatter(male_height, male_weight, color='blue')
    plt.scatter(female_height, female_weight, color='red')
    # plot the contours
    plt.contour(X, Y, male_qda, colors='b')
    plt.contour(X, Y, female_qda, colors='r')
    # plot the decision boundary
    boundary_qda = np.asarray(male_qda) - np.asarray(female_qda)
    plt.contour(X, Y, boundary_qda, 0, color='k')
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('contour and decision boundary for qda')
    plt.savefig("qda.pdf")
    plt.show()
    return (np.asarray(mu_male), np.asarray(mu_female), np.asarray(cov), np.asarray(cov_male), np.asarray(cov_female))


def misRate(mu_male, mu_female, cov, cov_male, cov_female, x, y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate

    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis

    x: a N-by-2 2D array contains the height/weight data of the N samples

    y: a N-by-1 1D array contains the labels of the N samples

    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    # lda
    male_lda = np.dot(mu_male.T, np.dot(np.linalg.inv(cov), x.T)) - 1 / 2 * np.dot(mu_male.T,
                                                                                   np.dot(np.linalg.inv(cov), mu_male))
    female_lda = np.dot(mu_female.T, np.dot(np.linalg.inv(cov), x.T)) - 1 / 2 * np.dot(mu_female.T,
                                                                                       np.dot(np.linalg.inv(cov),
                                                                                              mu_female))
    correct_lda = 0
    for i in range(0, len(y)):
        if (male_lda[i] >= female_lda[i] and y[i] == 1):
            correct_lda = correct_lda + 1
        elif (male_lda[i] <= female_lda[i] and y[i] == 2):
            correct_lda = correct_lda + 1
    mis_lda = 1 - correct_lda / len(y)
    print(mis_lda)

    # qda
    male_qda = []
    female_qda = []
    for i in range(0, x.shape[0]):
        male_qda.append(- 1 / 2 * np.log(np.linalg.det(cov_male)) - 1 / 2 * np.dot(x[i], np.dot(np.linalg.inv(cov_male),
                                                                                                x[i].T)) + np.dot(
            mu_male.T, np.dot(np.linalg.inv(cov_male), x[i].T)) - 1 / 2 * np.dot(mu_male.T,
                                                                                 np.dot(np.linalg.inv(cov_male),
                                                                                        mu_male)))
        female_qda.append(- 1 / 2 * np.log(np.linalg.det(cov_female)) - 1 / 2 * np.dot(x[i],
                                                                                       np.dot(np.linalg.inv(cov_female),
                                                                                              x[i].T)) + np.dot(
            mu_female.T, np.dot(np.linalg.inv(cov_female), x[i].T)) - 1 / 2 * np.dot(mu_female.T,
                                                                                     np.dot(np.linalg.inv(cov_female),
                                                                                            mu_female)))
    male_qda = np.asarray(male_qda)
    female_qda = np.asarray(female_qda)
    print(male_qda)
    print(female_qda)
    correct_qda = 0
    for i in range(0, len(y)):
        if (male_qda[i] >= female_qda[i] and y[i] == 1):
            correct_qda = correct_qda + 1
        elif (male_qda[i] <= female_qda[i] and y[i] == 2):
            correct_qda = correct_qda + 1
    mis_qda = 1 - correct_qda / len(y)
    print(mis_qda)

    return (mis_lda, mis_qda)

#
# if __name__ == '__main__':
#     # load training data and testing data
#     x_train, y_train = util.get_data_in_file('/Users/lijiani/Downloads/ldaqda/trainHeightWeight.txt')
#     x_test, y_test = util.get_data_in_file('/Users/lijiani/Downloads/ldaqda/testHeightWeight.txt')
#
#     # parameter estimation and visualization in LDA/QDA
#     mu_male, mu_female, cov, cov_male, cov_female = discrimAnalysis(x_train, y_train)
#
#     # misclassification rate computation
#     mis_LDA, mis_QDA = misRate(mu_male, mu_female, cov, cov_male, cov_female, x_test, y_test)
#

if __name__ == '__main__':

    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')

    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)

    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    print("lda is", mis_LDA)
    print("qda is", mis_QDA)





"""[176.78125, 71.875]
[129.25, 65.19117647058823]
[[12.859375, 58.16015625], [58.16015625, 1042.0458984375]]
[[9.772275086505186, 32.20220588235292], [32.20220588235292, 422.56985294117646]]"""