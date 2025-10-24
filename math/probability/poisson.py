#!/usr/bin/env python3
""" A script that represents a Poisson distribution """


class Poisson:
    def __init__(self, data=None, lambtha=1.):
        """
        A class with constructor that represents a Poisson distribution
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

    def cdf(self, k):
        """
        A function that calculates the CDF for a given number of successes
        """
        k = int(k)

        if k < 0:
            return 0

        e = 2.7182818285
        cdf_value = 0

        for i in range(k + 1):
            factorial = 1
            for j in range(1, i + 1):
                factorial *= j

            pmf = (e ** (-self.lambtha)) * (self.lambtha ** i) / factorial
            cdf_value += pmf

        return cdf_value
