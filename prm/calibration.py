import numpy as np
from prm.debug import ipsh

# #%%
# random_seed = 109

# # import the data and then divide into test and train datasets
# raw_data = pd.read_csv('data/breastcancer_binarized.csv')
# XY = raw_data.values
# Xf = XY[:, 0:XY.shape[1]]
# Y = XY[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(Xf,Y, test_size = 0.2, random_state = random_seed)

# # Train Baseline Logistic Regression Model
# clf = LogisticRegression()
# clf.fit(X_train, y_train)

# # Get Baseline Probabilities for Target Class (Y == 1)
# probabilities = clf.predict_proba(X_train)
# target_idx = np.flatnonzero(clf.classes_ == 1)
# probabilities = probabilities[:, target_idx]



def compute_calibration(y_true, y_score, n_bins = 10):
    """
        Compute calibration error for predicted probabilities based on observed y vals
        
        parameters
        ----------
        y_true: numpy array 
            y values observed
        y_score: numpy array
            predicted probabilities
            
        returns
        -------
        cal_error: float
            calibration metric for this classifier
        """
    #y[y == -1] = 0

    assert isinstance(n_bins, int) and n_bins >= 1

    binwidth = 1.0 / n_bins
    bin_left = np.arange(0, 1, step = binwidth)
    bin_right = bin_left + binwidth
    probs = y_score
    y = y_true

    predicted = []
    observed = []

    for k in range(n_bins):

        # find edges of kth bin
        l = bin_left[k]
        r = bin_right[k]

        # find points with predicted probability in bin
        # [0, 0.1)
        # [0.1, 0.2)
        # [0.2, 0.3)
        # ...
        # [0.9, 1.0) <- this should be [0.9, 1.0]
        if k < n_bins:
            binned_idx = np.logical_and(np.greater_equal(probs, l), np.less(probs, r))
        else:
            binned_idx = np.logical_and(np.greater_equal(probs, l), np.less_equal(probs, r))

        binned_idx = binned_idx.flatten()
        if any(binned_idx):
            predicted.append(np.mean(probs[binned_idx]))
            observed.append(np.mean(y[binned_idx]))

    observed = np.array(observed)
    predicted = np.array(predicted)
    cal_error = np.sqrt(np.square(observed - predicted).sum())
    
    # Uncomment below to plot calibration as its calculated
#     fig = plt.figure(figsize=(10, 10))
#     ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#     ax2 = plt.subplot2grid((3, 1), (2, 0))
    
#     ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#     ax1.plot(predicted, observed, "s-", label="Predicted")
#     ax1.set_ylabel("Fraction of positives")
#     ax1.set_ylim([-0.05, 1.05])
#     ax1.legend(loc="lower right")
#     ax1.set_title(f'Calibration plot')
    
#     ax2.hist(probs, range=(0, 1), bins=10, label="probs", histtype="step", lw=2)
#     ax2.set_xlabel("predicted value")
#     ax2.set_ylabel("Count")

    return cal_error

#cal_error = compute_calibration(y = y_train, probs = probabilities, n_bins = 10)