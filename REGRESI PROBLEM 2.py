import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

data = pd.read_csv("D:\KULIAH\Kuliah Semester 4 (2024)\METODE NUMERIK\Aplikasi Regresi\Student_Performance.csv")
NL = data['Sample Question Papers Practiced'].values
NT = data['Performance Index'].values

print(data)
NL = data['Sample Question Papers Practiced'].values
NT = data['Performance Index'].values

# Model Linear
linear_model = LinearRegression()
NL_reshaped = NL.reshape(-1, 1)
linear_model.fit(NL_reshaped, NT)
NT_pred_linear = linear_model.predict(NL_reshaped)
rms_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))

# Model Exponential
def exponential_model(x, a, b):
    return a * np.exp(b * x)

params, _ = curve_fit(exponential_model, NL, NT)
NT_pred_exponential = exponential_model(NL, *params)
rms_exponential = np.sqrt(mean_squared_error(NT, NT_pred_exponential))

# Model Polynomial
poly_coeffs = np.polyfit(NL, NT, 2)
NT_pred_poly = np.polyval(poly_coeffs, NL)
rms_poly = np.sqrt(mean_squared_error(NT, NT_pred_poly))

plt.figure(figsize=(14, 6))

plt.scatter(NL, NT, label='Data', color='black')

plt.plot(NL, NT_pred_linear, label='Linear Model', color='blue')

plt.plot(NL, NT_pred_exponential, label='Exponential Model', color='red')

plt.plot(NL, NT_pred_poly, label='Polynomial Model', color='green')

plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.title('Regresi Nilai Ujian Berdasarkan Jumlah Latihan Soal')
plt.show()

print(f'RMS Error Linear Model: {rms_linear}')
print(f'RMS Error Exponential Model: {rms_exponential}')
print(f'RMS Error Polynomial Model: {rms_poly}')


