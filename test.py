import  pandas,scipy,numpy
from sklearn.preprocessing import MinMaxScaler
df = pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',sep=";")
array = df.values
x = array[:,0:8]
y = array[:,8]
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(x)
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])

from  sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
rescaledX = scaler.transform(x)
print("Standardizing Data:")
print(rescaledX[0:5,:])

from  sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(x)
normalizedX = scaler.transform(x)
print("Normalization Data:")
print(normalizedX[0:5,:])

from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(x)
binaryX = binarizer.transform(x)
print("Binarizing Data:")
print(binaryX[0:5,:])

from sklearn.preprocessing import scale
data_standardized = scale(df)
print("Mean Removal:")
print(data_standardized.mean(axis=0))
print(data_standardized.std(axis=0))

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
print("One Hot Encoding:")
print(encoder.fit([[0,1,6,2],[1,5,3,5],[2,4,2,7],[1,0,4,2]]))
print(df.head())
print(df.describe())
import matplotlib.pyplot as plt
print("Correlation Matrix Plot in Visualizing Data:")
correlations = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()