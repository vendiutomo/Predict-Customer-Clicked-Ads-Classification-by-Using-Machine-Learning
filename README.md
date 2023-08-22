# Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning
#4 Mini Project at Rakamin Academy

In this project, machine learning modeling will be carried out, to predict potential users in digital advertising.

The business team wants to optimize its advertising methods on digital platforms in order to get potential users to click on a product. So that the cost to be incurred is not too high, create a machine learning model that detects potential users to convert or be interested in an advertisement so that we can optimize costs in advertising on digital platforms.

# Customer Type and Behaviour Analysis on Advertisement

## Univariate Analysis

![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/966323a7-5551-40f7-8a9c-60b11783875a)
> There are outliers in the Income Area feature.

![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/2e111a98-bb8e-4984-9aba-570d9e631bbb)
> 'Daily Time Spent on Site' and 'Daily Internet Usage' have a distribution that tends to approach the normal distribution.

> 'Income Area' has a negative skewed or left skew, meaning that more customers have low income than high income.

> 'Age' has a positive skewed or right skewed distribution, meaning that most customers are more than 30 years old compared to customers who are less than that.

## Bivariate Analysis

![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/09550d50-3643-4f4b-a28f-78c23267c8c6)
> The longer time spent on the site, customers are less likely to click on ads.

> The average age of those who click on ads is older (about 40 years) than those who don't (about 30 years).

> Customers who have large income tend not to click on ads.

> Active internet users tend to ignore ads.

![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/e5c60284-73b3-4992-b0eb-6a50d6647636)
> Customers who click on ads have a skew distribution to the right, meaning that the time spent on the site tends to be faster. Customers who don't click ads have a skew distribution to the left, meaning that their time spent on the site tends to be longer.

> The ages of customers who click on ads have a normal distribution. Meanwhile, the age of users who do not click on ads tends to be younger (around 30 years).

> The average income of customers who click on ads has a normal distribution. Meanwhile, users who do not click on ads tend to have high income (having a left skew distribution).

> The longer daily internet use, the lower the tendency for customers to click on ads. Meanwhile, the shorter the daily internet use, the higher the tendency of users to click ads.

![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/8a00a39e-f5e9-44a4-a7fe-88815c9e6d5a)
> The older the customer is, the shorter the time spent on the site and the shorter the internet usage, the more likely they are to click ads.

![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/97051e88-0fdf-4a21-bfe6-7fd151ff2f42)
> The less time spent on the website and the less internet usage, the more users tend to click on ads.

> The longer the time spent on the website and the more frequently the internet is used, the less likely users are to click on ads.

![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/5deb5fac-fcf9-4fd3-9041-fb80bf1e9bfa)
> Women are more likely to click ads than men.

> The three cities that have users clicking on ads more than those who don't click on them are Bandung, Surabaya and Bekasi respectively.

> West Java province has the highest number of users who click on ads compared to those who don't.

> The most categories that have a greater number of ad clicks than they do not click are automotive. Meanwhile, the health category is the highest category which has a greater number of unclicked ads than clicked.

## Multivariate Analysis

![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/9fb350b5-7a64-466d-a366-8a1bec3bf00d)
>*Daily Internet Usage* has a fairly high positive correlation with *Daily Time Spent on Site* (52%). Which means that the more often you use the internet, the longer the time spent on the website too.

> User age has a fairly high negative correlation with *Daily Internet Usage* (37%) and *Daily Time Spent on Site* (33%), where the older the age the shorter the daily internet usage and the shorter the time to open websites.

> Age also has a negative correlation with *Area Income* (18%), which means the older the age, the lower the income area.

# Data Cleaning & Preprocessing
- Missing Values
  - fill in the missing value in numerical features with median
  - fill in the missing value in categorical features with mode

- Outliers
  - handling outlier with IQR

- Feature Engineering
  - Feature Extraction
    - convert data type of Timestamp to datetime and extract it, the result is integer type data
  - Feature Encoding
    - Label Encoding
    - One-Hot Encoding
  - Feature Selection

- Split Data
  - Experiment 1 - without standardization (data train: 80%, data test: 20%)
  - Experiment 2 - with standardization (data train: 80%, data test: 20%)

# Data Modeling
## Modeling
    lr = LogisticRegression()
    nb = GaussianNB()
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    xgb = XGBClassifier()
    grad = GradientBoostingClassifier()
    ab = AdaBoostClassifier()
    LGBM = lgb.LGBMClassifier()

- Experiment 1
![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/3aa89f84-3f7c-4e1a-bca3-497fbcd6559d)

- Experiment 2
![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/83bba21e-01b6-411d-a271-408ea6585a95)

## Conclution
> Using recall and acuracy to compare the result

- Experiment 1
![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/014583ac-bccd-48ca-bc92-4f3b10b3f35e)

- Experiment 2
![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/088baea5-d5c0-4429-b1c9-22372e4de9c7)

> There is no significant difference in the performance of the train and test results, both without and with standardization.

> Most of the models have better performance after standardization, especially Logistic Regression, Naive Bayes, KNN which have significantly improved.

> From the modeling time, before standardization, the Decision Tree was the fastest, while after standardization it was also the Decision Tree.

> From the results, it can be concluded that the model used is a model with **LGBM after standardization** which has the highest recall value and accuracy compared to other models.

## Confision Matrix
![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/69b9b8b6-eb95-4d5d-b261-e67ed334344a)
> Of the 395 actual data that clicked ad on the data train, all of them were predicted to click ad.

> Of the 397 actual data that did not click on ad in the data train, all of them were predicted not to click on ad.

> It can be concluded that the model performance is very good because it predicts correctly.

![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/17e36443-3c7f-4940-a786-b9caeea07ffa)
> Of the 96 actual data that clicked ad on the test data, there were 93 data predicted to click ad and 3 data predicted not to click ad (should have clicked but predicted the opposite).

> Of the 103 actual data that did not click ad on the test data, there were 101 data predicted not to click ad and 2 data predicted to click ad (should not click but predicted the opposite).

> It was concluded that the model's performance was very good because it reduced the prediction error value in the actual data.

## Feature Importance
![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/44255c52-90c8-4411-8b91-d708effcd936)

The feature that most influences a user's decision to click on an ad (ad) is Daily Internet Usage, apart from that Area Income, Daily Time Spent on Site and Age are also quite influential.

- The lower the Daily Internet Usage, the more likely users are to click on ads, while the higher the Daily Internet Usage, the more likely users are not to click on ads.

- The higher the Income, the more inclined the user is not to click on ads, while the lower the Income, the more inclined the user is to click on ads.

- The lower the Daily Time Spent on Site, the more likely users are to click on ads, while the higher the Daily Time Spend on Site, the more likely users are not to click on ads.

- Users who are getting older are more likely to click on ads, while users who are younger are less likely to be interested in ads.

# Business Recomendation and Simulation

## Recomendation
Berdasarkan feature importance dan hasil EDA, dapat ditarik kesimpulan bahwa untuk meningkatkan ketertarikan customer dalam melihat iklan ada beberapa poin yang perlu diperhatikan:

- Perusahaan harus memerhatikan ketertarikan customer selama mereka menghabiskan waktu di website. Iklan yang ditampilkan adalah iklan yang sesuai dengan ketertarikan user tersebut.

- Penampilan iklan di website sebaiknya dikurangi untuk customer yang memiliki income di bawah 400 juta. Karena customer yang memiliki nominal income dibawah itu cenderung tidak mengklik iklan walaupun durasi yang dihabiskan dan penggunaan internet pada website tersebut tinggi.

- User dengan penggunaan internet harian dan waktu harian berkunjung pada website yang cukup tinggi, sebaiknya tidak diberikan iklan, kelompok ini cenderung mengabaikan iklan. Dengan ini dapat mengurangi beban biaya untuk menampilkan iklan.

- Customer dengan usia dibawah 35-38 tahun cenderung tidak tertarik dengan iklan, pada usia ini sebaiknya frekuensi munculnya iklan dikurangi. Hal ini dapat mengurangi beban biaya marketing.

- Agar iklan lebih tepat sasaran, dapat dilakukan modeling lebih lanjut seperti melakukan klustering custumer, agar dapat menampilkan iklan yang sesuai dengan karakteristik masing-masing user. Sehingga diharapkan akan lebih banyak yang mengklik ad dan kegiatan marketing ini akan menjadi lebih efektif.

## Simulation

- Before using machine learning:
![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/fc4f2593-629b-4b71-97ae-d606ff0e5aa3)

  - Biaya iklan (cost) = budget x n user = Rp500 x 1000 = Rp500.000
  - Conversion rate = 50% (500/1000 user clicked ad)
  - Revenue = price x n click = Rp2.000 x 500 = Rp1.000.000
  - Profit = Rp1.000.000 - Rp500.000 = Rp500.000

- After using machine learning:
![image](https://github.com/vendiutomo/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning/assets/128874036/32bab614-eb2e-439b-b9bc-a447eac8b7a8)

The conversion rate in the model used is 97%, meaning that out of 1000 users who receive advertisements, 970 of them click on advertisements.
So, at the same cost:
  - Revenue = IDR 2,000.00 * 970 = IDR 1,940,000.00
  - Profit = IDR 1,940,000.00 - IDR 500,000 = IDR 1,440,000.00

- Conclution:
  - In the simulation of this situation, if you do not use machine learning, the profit obtained is Rp. 500,000.00, while by using machine learning, the profit obtained will increase almost 3 times from the profit obtained previously.

  - From these results that machine learning can increase revenue and of course also profit from companies with quite significant results. 
