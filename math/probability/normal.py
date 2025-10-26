#!/usr/bin/env python3
""" A script that represents a Normal distribution """


class Normal:
    """ A class that represents a Normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Constructor for Normal distribution
        """

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        Returning the z-score of x
        """

        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """
        Returns the x-value of z
        """

        x = self.mean + z * self.stddev

        return x

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        pi = 3.1415926536
        e = 2.7182818285

        coefficient = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        pdf_value = coefficient * (e ** exponent)

        return pdf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        pi = 3.1415926536

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        term1 = z
        term2 = (z ** 3) / 3
        term3 = (z ** 5) / 10
        term4 = (z ** 7) / 42
        term5 = (z ** 9) / 216

        erf = (2 / (pi ** 0.5)) * (term1 - term2 + term3 - term4 + term5)

        cdf_value = 0.5 * (1 + erf)

        return cdf_value
