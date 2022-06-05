import numpy
from numpy.core.fromnumeric import var
import scipy.stats

def Cal_Probability_Distribution_Of_Gaussian_Distribution():
    location_parameter = float(input("Please input location parameter: "))
    variance = float(input("Please input variance: "))
    x = float(input("Please input x: "))
    res = scipy.stats.norm(location_parameter, variance).cdf(x)
    print(res)

def Cal_Quantile_Of_Gaussian_Distribution():
    location_parameter = float(input("Please input location parameter: "))
    variance = float(input("Please input variance: "))
    alpha = float(input("Please input alpha: "))
    res = scipy.stats.norm(location_parameter, variance).ppf(alpha)
    print(res)

def Cal_Quantile_Of_Chi_Square_Distribution():
    alpha = float(input("Please input alpha: "))
    degree_Of_freedom = int(input("Please input degree of freedom: "))
    res = scipy.stats.chi2.ppf(alpha, degree_Of_freedom)
    print(res)

def Cal_Quantile_Of_T_Distribution():
    alpha = float(input("Please input alpha: "))
    degree_Of_freedom = int(input("Please input degree of freedom: "))
    res = scipy.stats.t.ppf(alpha, degree_Of_freedom)
    print(res)

def Cal_Quantile_Of_F_Distribution():
    alpha = float(input("Please input alpha: "))
    degree_Of_freedom = int(input("Please input degree of freedom: "))
    res = scipy.stats.f.ppf(alpha, degree_Of_freedom)
    print(res)

choice = int(input("0: Gaussian Distribution\n1: Chi-Square Distribution\n2: T Distribution\n3: F Distribution\nPlease input a number to choose: "))
if (choice == 0):
    flag = int(input("0: Probability Distribution\n1: Quantile\nPlease input a number to choose: "))
    if (flag == 0):
        Cal_Probability_Distribution_Of_Gaussian_Distribution()
    elif (flag == 1):
        Cal_Quantile_Of_Gaussian_Distribution()
    else:
        print("Error!")
elif (choice == 1):
    Cal_Quantile_Of_Chi_Square_Distribution()
elif (choice == 2):
    Cal_Quantile_Of_T_Distribution()
elif (choice == 3):
    Cal_Quantile_Of_F_Distribution()
else:
    print("Error!")