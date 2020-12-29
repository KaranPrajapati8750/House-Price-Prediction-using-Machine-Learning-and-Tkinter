#importing libraries
import numpy as np
import pandas as pd
from tkinter import *
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#data from kc_house_data file
housing = pd.read_csv('house_data_25k.csv')

# splitting of data into train and test set on the basis of waterfront variable


split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2 , random_state=40)
for train_index,test_index in split.split(housing,housing['waterfront']):
    test_set = housing.loc[test_index]
    train_set = housing.loc[train_index]

#splitting of train data on dependent and independent variables
housing = train_set.copy()
housing = train_set.drop("price",axis=1)
housing_labels = train_set['price'].copy()

#performing pipeline to implement imputer and standardization on data
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing.drop(['id','date','yr_built','yr_renovated','zipcode'],axis=1))

#splitting of test data on dependent and independent variables
housing_test = test_set.copy()
housing_test = test_set.drop("price",axis=1)
housing_test_labels = test_set['price'].copy()

prepared_data = my_pipeline.transform(housing_test.drop(['id','date','yr_built','yr_renovated','zipcode'],axis=1))

#gradient boosting regressor
def gbr():

    from sklearn import ensemble

    clf1 = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
            learning_rate = 0.1, loss = 'ls')

    t = pd.DataFrame(columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','lat','long','sqft_living15','sqft_lot15'])
    t1 = t.append(pd.Series([Bedroom.get(),Bathroom.get(),Sqft_living.get(),Sqft_lot.get(),Floors.get(),Waterfront.get(),View.get(),Condition.get(),Grade.get(),Sqft_above.get(),Sqft_basement.get(),Lat.get(),Long.get(),Sqft_living15.get(),Sqft_lot15.get()],index=t.columns),ignore_index=True)
    t2 = my_pipeline.transform(t1)
    clf1.fit(housing_num_tr, housing_labels)
    s2 = clf1.score(prepared_data,housing_test_labels)
    print(s2)
    pred1 = clf1.predict(t2)
    p1.insert(0,str(pred1))


#decision tree regressor
def dtr():
    from sklearn.tree import DecisionTreeRegressor

    clf3 = DecisionTreeRegressor()

    t = pd.DataFrame(columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','lat','long','sqft_living15','sqft_lot15'])
    t1 = t.append(pd.Series([Bedroom.get(),Bathroom.get(),Sqft_living.get(),Sqft_lot.get(),Floors.get(),Waterfront.get(),View.get(),Condition.get(),Grade.get(),Sqft_above.get(),Sqft_basement.get(),Lat.get(),Long.get(),Sqft_living15.get(),Sqft_lot15.get()],index=t.columns),ignore_index=True)
    t2 = my_pipeline.transform(t1)
    clf3.fit(housing_num_tr, housing_labels)
    s3 = clf3.score(prepared_data,housing_test_labels)
    print(s3)
    pred3 = clf3.predict(t2)
    p3.insert(0,str(pred3))


#random forest regressor
def rfr():
    from sklearn.ensemble import RandomForestRegressor

    clf4 = RandomForestRegressor()

    t = pd.DataFrame(columns = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','lat','long','sqft_living15','sqft_lot15'])
    t1 = t.append(pd.Series([Bedroom.get(),Bathroom.get(),Sqft_living.get(),Sqft_lot.get(),Floors.get(),Waterfront.get(),View.get(),Condition.get(),Grade.get(),Sqft_above.get(),Sqft_basement.get(),Lat.get(),Long.get(),Sqft_living15.get(),Sqft_lot15.get()],index=t.columns),ignore_index=True)
    t2 = my_pipeline.transform(t1)
    clf4.fit(housing_num_tr, housing_labels)
    s4 = clf4.score(prepared_data,housing_test_labels)
    print(s4)
    pred4 = clf4.predict(t2)
    p4.insert(0,str(pred4))



#gui start
root = Tk()
root.configure(background='blue')
root.title('Project')

# entry variables
Bedroom = IntVar()
Bedroom.set(1)
Bathroom = IntVar()
Bathroom.set(1)
Floors = IntVar()
Floors.set(1)
Waterfront = IntVar()
Waterfront.set(0)
View = IntVar()
View.set(0)
Grade = IntVar()
Grade.set(3)
Sqft_living = IntVar()
Sqft_living.set(370)
Sqft_lot = IntVar()
Sqft_lot.set(520)
Sqft_above = IntVar()
Sqft_above.set(370)
Sqft_basement = IntVar()
Sqft_basement.set(0)
Condition = IntVar()
Condition.set(1)
Lat = IntVar()
Lat.set(47.15)
Long = IntVar()
Long.set(-122.51)
Sqft_living15  = IntVar()
Sqft_living15.set(399)
Sqft_lot15 = IntVar()
Sqft_lot15.set(651)

Name = StringVar()

# Heading
w2 = Label(root, justify=LEFT, text="House Predictor using Machine Learning", fg="white", bg="blue")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)
w2 = Label(root, justify=LEFT, text="A Project by Manoj and Karan", fg="white", bg="blue")
w2.config(font=("Aharoni", 30))
w2.grid(row=2, column=0, columnspan=2, padx=100)

# labels
NameLb = Label(root, text="Name", fg="yellow", bg="black")
NameLb.grid(row=6, column=0, pady=10, sticky=W)


S1Lb = Label(root, text="Bedroom from 1 to 33", fg="yellow", bg="black")
S1Lb.grid(row=7, column=0, pady=8, sticky=W)

S2Lb = Label(root, text="Bathroom from 1 to 8", fg="yellow", bg="black")
S2Lb.grid(row=8, column=0, pady=8, sticky=W)

S3Lb = Label(root, text="Sqft_living from 370 to 13540", fg="yellow", bg="black")
S3Lb.grid(row=9, column=0, pady=8, sticky=W)

S4Lb = Label(root, text="Sqft_lot from 520 to 1651359", fg="yellow", bg="black")
S4Lb.grid(row=10, column=0, pady=8, sticky=W)

S5Lb = Label(root, text="Floors from 1 to 4", fg="yellow", bg="black")
S5Lb.grid(row=11, column=0, pady=8, sticky=W)

S6Lb = Label(root, text="Waterfront from 0 to 1", fg="yellow", bg="black")
S6Lb.grid(row=12, column=0, pady=8, sticky=W)

S7Lb = Label(root, text="View from 0 to 4", fg="yellow", bg="black")
S7Lb.grid(row=13, column=0, pady=8, sticky=W)

S8Lb = Label(root, text="Condition from 1 to 5", fg="yellow", bg="black")
S8Lb.grid(row=14, column=0, pady=8, sticky=W)

S9Lb = Label(root, text="Grade from 3 to 13", fg="yellow", bg="black")
S9Lb.grid(row=15, column=0, pady=8, sticky=W)

S10Lb = Label(root, text="Sqft_above from 370 to 9410", fg="yellow", bg="black")
S10Lb.grid(row=16, column=0, pady=8, sticky=W)

S11Lb = Label(root, text="Sqft_basement from 0 to 4820", fg="yellow", bg="black")
S11Lb.grid(row=17, column=0, pady=8, sticky=W)

S12Lb = Label(root, text="Lat from 47.15 to 47.77", fg="yellow", bg="black")
S12Lb.grid(row=18, column=0, pady=8, sticky=W)

S13Lb = Label(root, text="Long from -122.51 to -121.31", fg="yellow", bg="black")
S13Lb.grid(row=19, column=0, pady=8, sticky=W)

S14Lb = Label(root, text="Sqft_living15 from 399 to 6210", fg="yellow", bg="black")
S14Lb.grid(row=20, column=0, pady=8, sticky=W)

S15Lb = Label(root, text="Sqft_lot15 from 651 to 872100", fg="yellow", bg="black")
S15Lb.grid(row=21, column=0, pady=8, sticky=W)

l1Lb = Label(root, text="GradientBoostingRegressor", fg="white", bg="red")
l1Lb.grid(row=24, column=0, pady=8,sticky=W)

l3Lb = Label(root, text="DecisionTreeRegressor", fg="white", bg="red")
l3Lb.grid(row=25, column=0, pady=8,sticky=W)

l4Lb = Label(root, text="RandomForestRegressor", fg="white", bg="red")
l4Lb.grid(row=26, column=0, pady=8,sticky=W)


# entries
NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

S1En = Spinbox(root, textvariable=Bedroom,from_=1,to=33)
S1En.grid(row=7, column=1)

S2En = Spinbox(root, textvariable=Bathroom,from_=1,to=8)
S2En.grid(row=8, column=1)

S3En = Spinbox(root, textvariable=Sqft_living,from_=370,to=13540)
S3En.grid(row=9, column=1)

S4En = Spinbox(root, textvariable=Sqft_lot,from_=520,to=1651359)
S4En.grid(row=10, column=1)

S5En = Spinbox(root, textvariable=Floors,from_=1,to=4)
S5En.grid(row=11, column=1)

S6En = Spinbox(root, textvariable=Waterfront,from_=0,to=1)
S6En.grid(row=12, column=1)

S7En = Spinbox(root, textvariable=View,from_=0,to=4)
S7En.grid(row=13, column=1)

S8En = Spinbox(root, textvariable=Condition,from_=1,to=5)
S8En.grid(row=14, column=1)

S9En = Spinbox(root, textvariable=Grade,from_=3,to=13)
S9En.grid(row=15, column=1)

S10En = Spinbox(root, textvariable=Sqft_above,from_=370,to=9410)
S10En.grid(row=16, column=1)

S11En = Spinbox(root, textvariable=Sqft_basement,from_=0,to=4820)
S11En.grid(row=17, column=1)

S12En = Spinbox(root, textvariable=Lat,from_=47.15,to=47.77)
S12En.grid(row=18, column=1)

S13En = Spinbox(root, textvariable=Long,from_=-122.51,to=-121.31)
S13En.grid(row=19, column=1)

S14En = Spinbox(root, textvariable=Sqft_living15,from_=399,to=6210)
S14En.grid(row=20, column=1)

S15En = Spinbox(root, textvariable=Sqft_lot15,from_=651,to=872100)
S15En.grid(row=21, column=1)


#button
b1 = Button(root, text="Price", command=gbr,bg="green",fg="yellow")
b1.grid(row=24, column=3,padx=10)


b3 = Button(root, text="Price", command=dtr,bg="green",fg="yellow")
b3.grid(row=25, column=3,padx=10)

b4 = Button(root, text="Price", command=rfr,bg="green",fg="yellow")
b4.grid(row=26, column=3,padx=10)

#predicted price
p1 = Entry(root)
p1.grid(row=24, column=1, padx=10)

p3 = Entry(root)
p3.grid(row=25, column=1, padx=10)

p4 = Entry(root)
p4.grid(row=26, column=1, padx=10)

root.mainloop()