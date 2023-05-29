# Ex-06-Feature-Transformation

# AIM

To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM

## STEP 1

Read the given Data

## STEP 2

Clean the Data Set using Data Cleaning Process

## STEP 3

Apply Feature Transformation techniques to all the features of the data set

## STEP 4

Save the data to the file

# CODE

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.isnull().sum()

df.describe()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

```

# OUTPUT

## DATASET

![Screenshot 2023-05-29 223847](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/6404fc01-6ffb-4d98-b0d4-d46693e79120)


## ISNULL

![Screenshot 2023-05-29 223855](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/2cffe03f-9ce9-439e-8c98-ec9f1feef8b1)


## INFO

![Screenshot 2023-05-29 223901](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/a5297899-9460-430c-b3b4-75e0a76f863a)


## DESCRIBE

![Screenshot 2023-05-29 223909](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/0c14fdfa-5b8a-4fd7-bdbb-a06a649df6f3)


## HIGHLY POSITIVE SKEW

![Screenshot 2023-05-29 224544](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/21c287d4-b514-4549-bb72-cf95a8919041)


## HIGHLY NEGATIVE SKEW

![Screenshot 2023-05-29 224549](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/3ee82cf6-3573-4e83-8671-a516b06d2157)


## MODERATE POSITIVE SKEW

![Screenshot 2023-05-29 224559](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/c79669db-5a61-421e-b9b6-5945200e445b)


## MODERATE NEGATIVE SKEW

![Screenshot 2023-05-29 224605](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/79d35dec-ded7-4973-8f15-f137b2d7f903)


## LOG OF MODERATE POSITIVE SKEW:

![Screenshot 2023-05-29 224615](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/b218222a-1371-46ed-9365-0978065e7834)


## LOG OF HIGHLY POSITIVE SKEW

![Screenshot 2023-05-29 224621](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/b249a2f7-6689-426f-b89a-adca6776ab95)


## RECIPROCAL OF HIGHLY POSITIVE SKEW
![Screenshot 2023-05-29 224627](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/3c3a5982-dd52-4ebb-bc79-48d6864e6202)


## SQUARE ROOT TRANSFORMATION
![Screenshot 2023-05-29 224640](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/afcc580d-39a0-4719-ae42-3b45969dee5b)



## POWER TRANSFORMATION OF MODERATE NEGATIVE SKEW

![Screenshot 2023-05-29 224646](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/1ecf944b-b301-4dc1-bb5a-79cc68ce1412)


## QUANTILE TRANSFORMATION

![Screenshot 2023-05-29 224652](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/c08a4139-feec-48d9-a1f1-aec71f4c1d84)
![Screenshot 2023-05-29 224658](https://github.com/Nagul71/Ex-06-Feature-Transformation/assets/118661118/758bf4ba-bf71-4e5b-9137-15b333162f51)


# RESULT:
Thus, Feature transformation is performed and executed successfully for the given dataset


