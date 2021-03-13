# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# import dataset
df = pd.read_csv("C:/Users/LENOVO/Desktop/PycharmProjects/bike_details.csv")

# get rows and dataset info
df.head() #top 5 row
df.info() #info dataset

# data preparation
for i in df.select_dtypes(include='object'):
    print(df[i].value_counts(), end='\n'*3)

df.replace({'1st owner':2, '2nd owner':2, '3rd owner':3, '4th owner':4}, inplace=True) #1st owner jadi 1
df.rename(columns = {'owner':'prev_owners'}, inplace=True) #table headaer "owner" jadi "prev_owner"

current_year = 2021
df['age'] = current_year - df['year'] #2021 - year di csv

df.drop(['year','name'], axis=1, inplace=True) #hapus kolom year & name

df_2 = df.copy() #variable baru df_2 copy hasil df
new_df = df.dropna().copy() #dataframe, hapus data yg null trus copy dari df
impute = SimpleImputer(strategy = 'median') #dari intrepreter sklearn, gtw funsignya

df_2_num = df_2.select_dtypes(exclude='object') #variable df_2_num ambil semua kolom kecuali datatype = object
df_2_cat = df_2['seller_type'] #variable df_2_cat ambil kolom seller_type
x = impute.fit_transform(df_2_num)
df_2 = pd.DataFrame(x, columns=df_2.select_dtypes(exclude='object').columns).copy()

df_2['seller_type'] = df_2_cat
new_df.corr() #hitung correlasi kolom

new_df_train_labels = new_df['selling_price_idr'] #variable ambil kolom selling_price_idr dari new_df
new_df.drop('selling_price_idr', axis=1, inplace=True) #new_df drop kolom selling_price_idr

num = new_df.select_dtypes(exclude='object').columns #variable num ambil smua kolom kecuali dtype = object
cat = new_df.select_dtypes(include='object').columns #variable num ambil smua kolom dengan dtype = object

# data preparation columntransformer
col_transformer = ColumnTransformer([('num',StandardScaler(), num),('cat', OneHotEncoder(), cat)]) #function dari sklearn
col_transformer

new_df_train_prepared = col_transformer.fit_transform(new_df) #jalankan fungsi column transformet.fittransform ke new_df
new_df_train_prepared

new_df_train_prepared = pd.DataFrame(new_df_train_prepared) #buat variable new_df_train_prepared menjadi dataframe hasil fit_transform
new_df_train_prepared

# divide attribute and labels
X = new_df_train_prepared.drop(4, axis=1) #ambil kolom 0 -> 3 dari new_df_train_prepared
y = new_df_train_prepared[4]#ambil kolom 4 dari new_df_train_prepared

# divide training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# train algorithm
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# predict data
y_pred = classifier.predict(X_test)

# confuse data
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))