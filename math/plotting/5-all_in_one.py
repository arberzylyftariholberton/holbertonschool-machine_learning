#!/usr/bin/env python3
""" A script that plots 5 graphics from the previos tasks """
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    A function that returns a figure with 5 graphics of previous tasks
    """

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # First Graphic
    all_plt = plt.figure()

    line_plt = all_plt.add_subplot(3, 2, 1)
    line_plt.plot(y0, color='red')
    line_plt.set_xlim((0, 10))

    # Second Graphic
    scatter_plt = all_plt.add_subplot(3, 2, 2)
    scatter_plt.scatter(x1, y1, color='magenta')
    scatter_plt.set_title("Men's Height vs Weight", fontsize='x-small')
    scatter_plt.set_xlabel("Height (in)", fontsize='x-small')
    scatter_plt.set_ylabel("Weight (lbs)", fontsize='x-small')

    # Third Graphic
    line2_plt = all_plt.add_subplot(3, 2, 3)
    line2_plt.plot(x2, y2)
    line2_plt.set_yscale("log")
    line2_plt.set_title("Exponential Decay of C-14", fontsize='x-small')
    line2_plt.set_xlabel("Time (years)", fontsize='x-small')
    line2_plt.set_ylabel("Fraction Remaining", fontsize='x-small')
    line2_plt.set_xlim(x2[0], x2[-1])

    # Fourth Graphic
    exp_plt = all_plt.add_subplot(3, 2, 4)
    exp_plt.plot(x3, y31, color='red', linestyle='dashed', label='C-14')
    exp_plt.plot(x3, y32, color='green', label='Ra-226')
    exp_plt.set_title(
        "Exponential Decay of Radioactive Elements",
        fontsize='x-small')
    exp_plt.set_xlabel("Time (years)", fontsize='x-small')
    exp_plt.set_ylabel("Fraction Remaining", fontsize='x-small')
    exp_plt.set_xlim((0, 20000))
    exp_plt.set_ylim((0, 1))
    plt.legend(fontsize='x-small')

    # Fifth Graphic
    hist_plt = all_plt.add_subplot(3, 1, 3)
    bins = np.arange(0, 101, 10)
    hist_plt.hist(student_grades, bins=bins, range=(0, 100), edgecolor='black')
    hist_plt.set_title("Project A", fontsize='x-small')
    hist_plt.set_xlabel("Grades", fontsize='x-small')
    hist_plt.set_ylabel("Number of Students", fontsize='x-small')
    hist_plt.set_xlim(0, 100)
    hist_plt.set_ylim(0, 30)
    hist_plt.set_xticks(np.arange(0, 101, step=10))
    hist_plt.set_yticks(np.arange(0, 31, 10))

    plt.suptitle("All in One")
    plt.tight_layout()
    plt.show()
