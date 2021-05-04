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

    reshaped_x = np.transpose(x)
    reshaped_x = np.vstack((reshaped_x[0], reshaped_x[1], y))
    males_data = [[] for i in range(3)]
    females_data = [[] for i in range(3)]
    males_count = 0
    females_count = 0
    for i in range(len(reshaped_x[0])):
        if reshaped_x[2][i] == 1:
            males_count += 1
            males_data[0].append(reshaped_x[0][i])
            males_data[1].append(reshaped_x[1][i])
            males_data[2].append(1)
        elif reshaped_x[2][i] == 2:
            females_count += 1
            females_data[0].append(reshaped_x[0][i])
            females_data[1].append(reshaped_x[1][i])
            females_data[2].append(2)


    # calculate means
    mu_male = [np.mean(males_data[1]), np.mean(males_data[0])]
    mu_female = [np.mean(females_data[1]), np.mean(females_data[0])]
    mu_height = np.mean(reshaped_x[1])
    mu_weight = np.mean(reshaped_x[0])

    # calculate total covariance
    cov_00 = 0
    cov_01 = 0
    cov_11 = 0
    for i in range(len(y)):
        cov_00 = cov_00 + (x[i][0] - mu_weight) ** 2
        cov_01 = cov_01 + (x[i][1] - mu_height) * (x[i][0] - mu_weight)
        cov_11 = cov_11 + (x[i][1] - mu_height) ** 2

    cov = [[cov_00 / len(y), cov_01 / len(y)], [cov_01 / len(y), cov_11 / len(y)]]

    # calculate covariance for male and female
    #mu_male: height, weight
    #mu_female: height, weight
    # x[item]: weight, height
    cov_male_00 = 0
    cov_male_01 = 0
    cov_male_11 = 0
    cov_female_00 = 0
    cov_female_01 = 0
    cov_female_11 = 0

    for item in range(len(y)):
        if y[item] == 1:
            cov_male_00 += (x[item][0] - mu_male[1]) ** 2
            cov_male_01 += (x[item][0] - mu_male[1]) * (x[item][1] - mu_male[0])
            cov_male_11 += (x[item][1] - mu_male[0]) ** 2
        else:
            cov_female_00 += (x[item][0] - mu_female[1]) ** 2
            cov_female_01 += (x[item][0] - mu_female[1]) * (x[item][1] - mu_female[0])
            cov_female_11 += (x[item][1] - mu_female[0]) ** 2

    cov_male = [[cov_male_00 / males_count, cov_male_01 / males_count],
                [cov_male_01 / males_count, cov_male_11 / males_count]]
    cov_female = [[cov_female_00 / females_count, cov_female_01 / females_count],
                  [cov_female_01 / females_count, cov_female_11 / females_count]]


    mu_male.reverse()
    mu_female.reverse()
    # lda plot
    # plot all datapoints
    # males data, height, weight
    plt.scatter(males_data[0], males_data[1], color = 'blue')
    plt.scatter(females_data[0], females_data[1], color = 'red')
    #print(males_data)
    # plot gradient
    X, Y = np.meshgrid(np.linspace(50, 80, 100), np.linspace(80, 280, 100))
    male_lda = []
    female_lda = []
    male_qda = []
    female_qda = []
    xx = X[0].reshape(100, 1)
    # make data points samples to pass into density_Gaussian
    for i in range(0, 100):
        samples = np.concatenate((xx, Y[i].reshape(100, 1)), 1)
        male_lda.append(util.density_Gaussian(mu_male, cov, samples))
        female_lda.append(util.density_Gaussian(mu_female, cov, samples))
        male_qda.append(util.density_Gaussian(mu_male, cov_male, samples))
        female_qda.append(util.density_Gaussian(mu_female, cov_female, samples))

    # plot the contours
    plt.contour(X, Y, male_lda, colors = 'b')
    plt.contour(X, Y, female_lda, colors = 'r')

    # plot the decision boundary
    boundary_lda = np.asarray(male_lda) - np.asarray(female_lda)
    plt.contour(X, Y, boundary_lda, 0, color = 'k')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Contour and decision boundary for LDA')
    plt.savefig("lda.pdf")
    plt.show()

    # qda plot
    plt.scatter(males_data[0], males_data[1], color = 'blue')
    plt.scatter(females_data[0], females_data[1], color='red')

    plt.contour(X, Y, male_qda, colors = 'b')
    plt.contour(X, Y, female_qda, colors = 'r')
    boundary_qda = np.asarray(male_qda) - np.asarray(female_qda)
    plt.contour(X, Y, boundary_qda, 0, colors = 'k')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Contour and decision boundary for QDA')
    plt.savefig("qda.pdf")
    plt.show()

    return (mu_male,mu_female,cov,cov_male,cov_female)


def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
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
    # LDA
    cov_inv = np.linalg.inv(cov)
    mu_male_trans = np.transpose(mu_male)
    mu_female_trans = np.transpose(mu_female)

    male_lda = np.dot(np.dot(x, cov_inv), mu_male)
    male_lda -= 0.5 * np.dot(np.dot(mu_male_trans, cov_inv), mu_male_trans)

    female_lda = np.dot(np.dot(x, cov_inv), mu_female)
    female_lda -= 0.5 * np.dot(np.dot(mu_female_trans, cov_inv), mu_female_trans)

    correct_lad = 0
    for i in range(len(y)):
        if male_lda[i] >= female_lda[i] and y[i] == 1:
            correct_lad += 1
        elif male_lda[i] <= female_lda[i] and y[i] == 2:
            correct_lad += 1
    mis_lda = 1-correct_lad / len(y)

    # QDA
    correct_qda = 0
    cov_male_det = np.linalg.det(cov_male)
    cov_female_det = np.linalg.det(cov_female)
    cov_male_inv = np.linalg.inv(cov_male)
    cov_female_inv = np.linalg.inv(cov_female)

    for i in range(len(x)):
        m_qda = -0.5 * np.log(cov_male_det) - 0.5 * np.dot(np.dot((x[i] - mu_male).T, cov_male_inv), (x[i] - mu_male))
        f_qda = -0.5 * np.log(cov_female_det) - 0.5 * np.dot(np.dot((x[i] - mu_female).T, cov_female_inv), (x[i] - mu_female))
        if m_qda >= f_qda and y[i] == 1:
            correct_qda = correct_qda + 1
        elif m_qda <= f_qda and y[i] == 2:
            correct_qda = correct_qda + 1

    mis_qda = 1 - correct_qda / len(y)
    return (mis_lda, mis_qda)


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





