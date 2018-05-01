from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np


# Bu projede Sınıflandırma Algoritmalarından, Cart algoritmasini kullanan; scikitlearn kütüphanesinden
# DecisionTreeClassifier kullanarak sınıflandırma yaptım.

dataset = pd.read_csv("data/adult.csv")    #dataset yüklendi.
#print(dataset.head(10))

####### Veriyi Tanıma ##############################################

sayac = 0                                  		   #Verideki "<=50" olan kayıt sayısı
for i in dataset["income"]:
    if i == ">50K":
        sayac = sayac + 1

sayac2 = len(dataset["income"]) - sayac                     #Verideki ">50" olan kayıt sayısı
print("Verisetinde 50K'dan yüksek gelire sahip",sayac,
      ", 50K'dan düşük gelire sahip", sayac2 ,"veri vardır.\n\n")

#Sınıf etiketleri grafiği çizdirildi.
x = np.arange(2)
degerler = [sayac, sayac2]
fig, ax = plt.subplots()
plt.title("Etiket Durumu")
plt.bar(x, degerler)
plt.xticks(x, ('>50K', '<=50'))
plt.show()


####### Veri Temizleme ##############################################

#Cinsiyet degerleri iki değerli kategorik veri haline getirildi. Kadın(1) Erkek(0)
dataset["sex"] = dataset["sex"].map({"Male": 0, "Female":1})

#Marial-Status kolonunun ayrık değerleri toparlanarak, iki degerli kategorik veri haline getirildi. Yes(Evli)(1) No(EvliDeğil)(0)
dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
dataset["marital.status"] = dataset["marital.status"].map({"Married":1, "Single":0})
dataset["marital.status"] = dataset["marital.status"].astype(int)

#Aşağıdaki her kolonda da ayrı ayrı, ' ValueError: could not convert string to float ' hatası aldım.
#Bu hatayı uzun süre uğraşıp çözemediğim için bu kolonları kaldırdım.
dataset.drop(labels=["workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)
print("Verinin ilk 5 satırı aşağıda gösterilmiştir.\n")
print(dataset.head(5))

####### Ağaç oluşturma ##############################################


X = dataset.values[:, 0:8]   #diğer kolonlar
Y = dataset.values[:, 8:9]   #hedef kolonum (sınıf etiketi)


#Verinin %30 u test verisi olarak ayrıldı.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

decisionTree = tree.DecisionTreeClassifier(max_depth = 5) #max derinliği 5 olacak şekilde karar ağacı olusturuldu.
decisionTree.fit(X_train, y_train)    #eğitim verisi ile model oturtuldu.


treePredictionResult = decisionTree.predict(X_test)  #ağac üzerinden tahmin yapıldı.

#dogru ve yanlış tahmin edilen kayıt sayıları bulundu.
dogru = accuracy_score(y_test,treePredictionResult,normalize=False)
yanlis = len(y_test)-accuracy_score(y_test,treePredictionResult,normalize=False)
dogru_oran = accuracy_score(y_test,treePredictionResult)*100
yanlis_oran = ((len(y_test)-accuracy_score(y_test,treePredictionResult,normalize=False))/(len(y_test))*100)

print(" \n\nDoğru sınıflandırılan test verisi sayısı :",dogru,", oranı: %",dogru_oran)
print("Yanlış sınıflandırılan test verisi sayısı :",yanlis,", oranı: %",yanlis_oran)

#accuracy score bulundu.
tree_accuracy = decisionTree.score(X_test,y_test)
print("Ağacın Accuracy Score Değeri :", tree_accuracy,"\n\n")


#dogru ve yanlıs verilerin sayısını gosteren grafik çizdirildi.
x = np.arange(2)
degerler = [dogru, yanlis]
fig, ax = plt.subplots()
plt.title("Model Başarısı")
plt.bar(x, degerler)
plt.xticks(x, ('Dogru', 'Yanlıs'))
plt.show()



#Verilerin birbirleriyle ilişkisini açıklayan,Korelasyon Matrisi çizdirildi.
f,ax = plt.subplots(figsize=(10, 10))
plt.title("Korelasyon Matrisi")
sns.heatmap(dataset.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)



#Özelliklerin önemini Gösteren Grafik çizdirildi.
feature_important = decisionTree.feature_importances_
#print(decisionTree.feature_importances_)

pos = np.arange(8) + 0.5
fig, ax = plt.subplots(figsize=(13, 6))
plt.barh(pos, feature_important, align='center')
plt.title("Özellik Önemi")
plt.xlabel("Model Accuracy")
plt.ylabel("Features")
plt.yticks(pos, ('Age', 'Fnlwgt','Education Num', 'Marital Status','Sex', 'Capital Gain', 'Capital Loss','Hours Per Week'))
plt.grid(True)
plt.show()



