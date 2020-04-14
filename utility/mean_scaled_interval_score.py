def indicator_function(larger, lower):
    if larger > lower:
        return 1
    else:
        return 0


def mean_scaled_interval_score(y, u, l, alpha=0.05, h=1, m=12):
    """

    :param y: Real values
    :param u: Upper bound of prediction interval
    :param l: Lower bound of prediction interval
    :param alpha: Significance level
    :param h: Forecast horizon of the h-step forecast
    :param m: Seasonal frequency
    :return: mean scaled interval score
    """
    if h > 1:
        interval_score = 0
        for i in range(h):
            interval_score += u[i]-l[i] + 2/alpha * (l[i]-y[i]) * indicator_function(larger=l[i], lower=y[i]) \
                              + 2/alpha * (y[i] - u[i]) * indicator_function(larger=y[i], lower=u[i])
        # deltam_y = 0
        # for i in range()
        return interval_score/h
    else:
        interval_score = u - l + 2 / alpha * (l - y) * indicator_function(larger=l, lower=y) \
                          + 2 / alpha * (y - u) * indicator_function(larger=y, lower=u)
        return interval_score
