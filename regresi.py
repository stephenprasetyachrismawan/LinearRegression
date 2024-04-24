import pandas as pd
import matplotlib.pyplot as  plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score



df = pd.read_csv("DailyDelhiClimateTrain.csv")  # Load the dataset
plt.scatter(df.humidity,df.meantemp,color="blue")
plt.xlabel("Humidity")
plt.ylabel("Mean Temperature")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]
print(f"\nSelamat DATANG. Aplikasi Regresi untuk menghitung prediksi Temperatur Rata-Rata berdasarkan nilai Humidity ")
#inputan User = 


def mean_absolute_error(y, y_predict):
    n = len(y)
    res = 0
    for i in range(n):
        res += abs(y[i] - y_predict[i])
    return res / n

def mean_absolute_precentage_error(y, y_predict):
    n = len(y)
    res = 0
    for i in range(n):
        res += abs((y[i] - y_predict[i])/y[i])
    return res / n

    
    
#membuat model regresi
regr = LinearRegression()
train_x = np.asanyarray(train[['humidity']])
train_y = np.asanyarray(train[['meantemp']])
regr.fit(train_x, train_y)
train_y_predict = regr.predict(train_x)


plt.scatter(train.humidity, train.meantemp,color="blue")
plt.scatter(train_x, train_y_predict, color="green", label="Prediksi Model")
plt.plot(train_x,regr.coef_[0][0]*train_x+regr.intercept_[0],'-r')
plt.xlabel("Humidity")
plt.ylabel("Mean Temp")
plt.title("Regresi Linear Data Train")
plt.show()


maeTrain = mean_absolute_error(train_y,train_y_predict)
mapeTrain =mean_absolute_precentage_error(train_y,train_y_predict)*100
print(f"\nMAE Data TRAIN : {maeTrain[0]}")
print(f"MAPE Data TRAIN : {mapeTrain[0]}%")
print(f"Dengan R2Score/Akurasi data TRAIN = {r2_score(train_y,train_y_predict)}")

test_x = np.asanyarray(test[['humidity']])
test_y = np.asanyarray(test[['meantemp']])
test_y_predict = regr.predict(test_x)




plt.scatter(test.humidity, test.meantemp,color="blue")
plt.scatter(test_x, test_y_predict,color="green")
plt.plot(test_x,regr.coef_[0][0]*test_x+regr.intercept_[0],'-r')
plt.xlabel("Humidity")
plt.ylabel("Mean Temp")
plt.title("Regresi Linear Data Test")
plt.show()


maeTest = mean_absolute_error(test_y,test_y_predict)
mapeTest = mean_absolute_precentage_error(test_y,test_y_predict)*100
print(f"\nMAE Data TEST : {maeTest[0]}")
print(f"MAPE Data TEST : {mapeTest[0]}%")
print(f"Dengan R2Score/Akurasi data TEST = {r2_score(test_y,test_y_predict)}")

#inputan User = 

ulang = "a"
while ulang == "y" or "a" :
    def input_angka():
        while True:
            user_input = input("\nUntuk memprediksi temperatur,  Masukkan nilai Humidity: ")
            # Periksa apakah setiap karakter adalah angka atau koma
            if all(char.isdigit() or char == '.' for char in user_input):
                return float(user_input)  # Kembalikan input sebagai float
            else:
                print("\nInput tidak valid. Mohon masukkan angka saja.")

    # Contoh penggunaan
    x_user = input_angka()
    x_user = np.array([[x_user]])
    user_y_predict = regr.predict(x_user)
    
    print(f"\nHasil Prediksi Data Humidity = {x_user[0][0]} , Rata Rata Temperature adalah = {user_y_predict[0][0]}")

    
    ulang = input("Ingin Mengulangi Prediksi? Y/N \n").lower()
    if ulang == "n" :
        break
    