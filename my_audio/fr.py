import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

reference_db = 1.2
fr = pd.read_csv("response_original.csv", sep=r'\s*,\s*')


def interp_log(x, y):
    interp = CubicSpline(np.log10(x / 2), y)
    def f(x):
        return interp(np.log10(x / 2))
    return f


if __name__ == "__main__":
    x = np.linspace(20, 20000, 4000)
    fr_interp = interp_log(fr["freq"], fr["gain"] - reference_db)

    plt.scatter(fr["freq"], -(fr["gain"] - reference_db))
    plt.semilogx(x, -fr_interp(x))
    plt.plot([20, 20000], [0, 0])
    plt.plot([20, 20000], [-12, -12])
    plt.xlim(20, 20000)
    plt.show()
