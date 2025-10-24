#!/usr/bin/env python3
""" A script that represents a Poisson distribution """


class Poisson:
    """ A class with constructor that represents a Poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """
        Function with constructor that returns a Poisson distribution
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the PMF for a given number of successes
        Return PMF value for k
        """
        k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        pmf_successes = (e ** (-self.lambtha))*(self.lambtha ** k) / factorial

        return pmf_successes

    def cdf(self, k):
        """
        A function that calculates the CDF for a given number of successes
        """
        k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285
        cdf_successes = 0

        for i in range(k + 1):
            factorial = 1
            for j in range(1, i + 1):
                factorial *= j

            pmf = (e ** (-self.lambtha)) * (self.lambtha ** i) / factorial
            cdf_successes += pmf

        return cdf_successes
