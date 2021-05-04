import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

total_words_spam_global = 0
word_list_spam_global = 0
total_words_ham_global = 0
word_list_ham_global = 0
total_words = 0

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set

    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the
    smoothed estimates of q_d
    """
    ### TODO: Write your code here
    # all_words_dict = util.get_counts(file_lists_by_category[0] + file_lists_by_category[1])
    # global total_words
    # unique_words = len(all_words_dict)
    # spam_word_dict = util.get_word_freq(file_lists_by_category[0])
    # global total_words_spam_global
    # for word in spam_word_dict:
    #     total_words_spam_global += spam_word_dict[word]
    # for word in spam_word_dict:
    #     spam_word_dict[word] = (spam_word_dict[word]+1)/(total_words_spam_global+total_words)
    # ham_word_dict = util.get_word_freq(file_lists_by_category[1])
    # global total_words_ham_global
    # for word in ham_word_dict:
    #     total_words_ham_global += ham_word_dict[word]
    # for word in ham_word_dict:
    #     ham_word_dict[word] = (ham_word_dict[word]+1)/(total_words_ham_global+total_words)
    #
    # print("the thing is ", total_words_spam_global, total_words_ham_global, total_words)
    # probabilities_by_category = (spam_word_dict, ham_word_dict)



    SPAM_files = file_lists_by_category[0]
    HAM_files = file_lists_by_category[1]

    word_list_spam = util.get_word_freq(SPAM_files)
    word_list_ham = util.get_word_freq(HAM_files)
    word_list_total = set(util.get_word_freq(SPAM_files + HAM_files))

    global total_words_spam_global
    global total_words_ham_global
    for word in word_list_spam:
        total_words_spam_global += word_list_spam[word]
    for word in word_list_ham:
        total_words_ham_global += word_list_ham[word]

    global total_words
    total_words = len(word_list_total)

    for word in word_list_spam:
        word_list_spam[word] = (word_list_spam[word] + 1) / (total_words_spam_global + total_words)

    for word in word_list_ham:
        word_list_ham[word] = (word_list_ham[word] + 1) / (total_words_ham_global + total_words)
    print("the thing is ", total_words_spam_global, total_words_ham_global, total_words)

    # for word in word_list_spam:
    #     if word not in word_list_ham:
    #         word_list_ham[word] = 1 / (total_words_ham_global + total_words)
    #
    # for word in word_list_ham:
    #     if word not in word_list_spam:
    #         word_list_spam[word] = 1 / (total_words_spam_global + total_words)

    probabilities_by_category = tuple((word_list_spam, word_list_ham))


    return probabilities_by_category

# def get_words_in_file(filename):
#     """ Returns a list of all words in the file at filename. """
#     with open(filename, 'r', encoding = "ISO-8859-1") as f:
#         # read() reads in a string from a file pointer, and split() splits a
#         # string into words based on whitespace
#         words = f.read().split()
#     return words
#
# def get_files_in_folder(folder):
#     """ Returns a list of files in folder (including the path to the file) """
#     filenames = os.listdir(folder)
#     # os.path.join combines paths while dealing with /s and \s appropriately
#     full_filenames = [os.path.join(folder, filename) for filename in filenames]
#     return full_filenames
#
# not needed for now
# def get_counts(file_list):
#     """
#     Returns a dict whose keys are words and whose values are the number of
#     files in file_list the key occurred in.
#     """
#     counts = Counter()
#     for f in file_list:
#         words = get_words_in_file(f)
#         for w in set(words):
#             counts[w] += 1
#     return counts
#
# def get_word_freq(file_list):
#     """
#     Returns a dict whose keys are words and whose values are word freq
#     """
#     counts = Counter()
#     for f in file_list:
#         words = get_words_in_file(f)
#         for w in words:
#             counts[w] += 1
#     return counts

def classify_new_email(filename,probabilities_by_category,prior_by_category, c):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified

    probabilities_by_category: output of function learn_distributions

    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)],
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    word_list = util.get_words_in_file(filename)

    bayes_spam = 1
    bayes_ham = 1

    distribution_spam = probabilities_by_category[0]
    distribution_ham = probabilities_by_category[1]
    for word in word_list:
        if word in distribution_spam:
            bayes_spam = bayes_spam + np.log(distribution_spam[word])
        else:
            bayes_spam = bayes_spam - np.log(total_words_spam_global + total_words)

        if word in distribution_ham:
            bayes_ham = bayes_ham + np.log(distribution_ham[word])
        else:
            bayes_ham = bayes_ham - np.log(total_words_ham_global + total_words)

    bayes_spam += np.log(prior_by_category[0])
    bayes_ham += np.log(prior_by_category[1])
        #bayes_ham = bayes_ham + np.log(distribution_ham[word])
    #print(bayes_spam, bayes_ham)

    str = ""
    # if c== 0.000000001:
    #     print(bayes_spam, bayes_ham)
    if bayes_spam - bayes_ham >= np.log(c):
        str = "spam"
    else:
        str = "ham"

    li = list((bayes_spam, bayes_ham))
    classify_result = [str, li]


    return classify_result

if __name__ == '__main__':

    # folder for training and testing
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))

    # Learn the distributions
    probabilities_by_category = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # Store the classification results



    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam'
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham'
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam'
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham'

    # Classify emails from testing set and measure the performance
    C = [0,1e-18, 1e-10, 1e-5, 1e-2, 1, 1e2, 1e4, 1e8, 1e13, 1e26]
    error1 = []
    error2 = []
    for c in C:
        performance_measures = np.zeros([2, 2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                     probabilities_by_category,
                                                     priors_by_category, c)
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0],totals[0],correct[1],totals[1]))
        error1.append(totals[0] - correct[0])
        error2.append(totals[1] - correct[1])

    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    print(error1)
    print(error2)
    plt.plot(error1, error2)
    plt.xlabel('type 1 errors')
    plt.ylabel('type 2 errors')
    plt.title('trade off of type 1 error and type 2 errpr')
    plt.savefig("nbc.pdf")
    plt.show()
