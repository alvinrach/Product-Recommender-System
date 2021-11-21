# Product Recommendation System

## Project Overview
The recommendation system is a form of machine learning application. There are many types of recommendation systems, one of which is a product recommendation system. One business that uses a product recommendation system is the e-commerce sector. Product recommendations on e-commerce itself are important because they can improve business performance from e-commerce.

[Recommendation system on e-commerce](https://www.ismll.uni-hildesheim.de/pub/pdfs/Hauger_Tso_Schmidt-Thieme_GFKL2007.pdf) is implemented to provide appropriate product recommendations to customers for certain conditions. For example, Shopee provides recommendations after customers buy certain items.

[Multiple techniques](https://d1wqtxts1xzle7.cloudfront.net/59762468/10.1.1.695.642820190617-91457-z4s1rf-with-cover-page-v2.pdf?Expires=1636075790&Signature=GHF61q6wBl4xM2--demA-QMVbt8M1m~xlwaxuWZzvZucb7Fm51fn8zxmlu-vmtY0onvuECBIOkJzdpQqOFXl6kd5DNFQpcBGU5IGkvweabciuSfGph7esQY7ppLZnEKDqZGJMes5Oxrx6FCLBVTM3MSC5bBZB3~WStLKxDBA40l1hwfnpFAm1bpilZTWams3HVVHnPW~5LFXipg2MAYlTMQAjmOQBfSCChyezzmzr2kV79qs8jMosE5s-zWR5acQxn3Z7Ey4M5orybysI3BbVUK~RfZJq6hNy7a1ZHFgJKsTOnxy5VsaU5mBLEDHFouyiacmxg0Tsmgzr1bCVfRvww__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) used in the system following recommendations are content-based filtering and collaborative fitering. There is also a hybrid recommendation system, which uses a combination of several recommended filtering systems. In this project, content-based filtering and collaborative filtering will be used to build a product recommendation system.

## Business Understanding
### Problem Statements
1. How to issue product recommendations from a particular dataset?
2. How to perform metric measurements on the product recommendation system?

### Goals
1. Issue product recommendations using content-based filtering and collaborative filtering
2. Take measurements using certain metrics in the product recommendation system

### Solution Approach
#### Content-Based Filtering
Content-based filtering recommends items similar to other items in the past. For example, if someone likes dolls, the system also recommends those that have similar feature values to dolls. Dolls are categorized as women's goods, so women's goods will also be recommended to users who buy dolls. What has just been mentioned is just an example, it can also involve other features to create the recommendation system.

The advantage of this system is that it can be used for new customers. This is because new customer data usually does not have rating data. Therefore content-based filtering can be run with very little data that has been entered in the sample. While the drawback of this measurement is that it is not easy to carry out metric measurements.

The principle of this system is to measure the similarity score of each item with every other item, and then order the similarity scores. The items with the highest similarity scores will be displayed as recommendation results.

For input processing, if there is a review in the form of a sentence, then TfIdfVectorizer or CountVectorizer is used. And if there are categorical features, one-hot encoding is used. Furthermore, the similarity measurement uses several algorithms, one of which uses cosine_similarity.

#### Collaborative Filtering
Collaborative filtering recommends items based on user ratings of items. This system is divided into memory-based and model-based. Memory-based itself is further divided into user-based and item-based. Meanwhile, model-based is divided into cluster-based, matrix factorization and deep learning.

User-based searches for users with similar tastes, for example, they both like comedy films. Item-based calculates the similarity between each item from all users. Cluster-based uses a clustering algorithm (K-Means, Gaussian Mixture Model, DBScan) to group users according to available variables. Matrix Factorization uses matrix decomposition formation. Meanwhile systems with deep learning use deep learning algorithms to determine the final recommendation.

In the product recommendation system approach by collaborative filtering using the embedding technique, this time user data, item and user ratings will be used for the item. Each user and item will be embedded to be modeled with the target, namely the rating itself.

The advantage of collaborative filtering is that it is not difficult to measure metrics. While the drawback is that it cannot be used if there is no user rating/rating data for the item.

## Data Understanding
[Data source](https://www.kaggle.com/ruchi798/marketing-bias-in-product-recommendations) used is from Kaggle. In source code, data is pulled using the [Kaggle API](https://www.kaggle.com/docs/api#interacting-with-datasets). To retrieve data manually, it is done by downloading directly from the data source link. The dataset used has 1292954 samples/rows non-duplication.

There are several variables/features in the dataset, but the ones that will be used in this project are:
- item_id : The product id. Because the dataset does not have a product name, item_id will be used as a product name pointer. There are 9560 types of products.
- user_id : User id. There are 1157633 unique users.
- Rating: The rating given by the customer when the order is completed.
- model_attr : Product model, whether the product is intended for women, men, or both.
- category : Category of the product. There are 10 types of categories.
- brand : Brand of the product. Some products do not have a brand, so they will be labeled unbranded using fillna.
- year : The year of manufacture of the product.

What will be used in content-based filtering are item_id, category, brand and year. Ratings are not used because it assumes content-based filtering this time is intended for businesses that have not collected a fairly complete dataset of ratings, so data other than ratings are used.

Meanwhile, what will be used for collaborative filtering are item_id, user_id and rating. Collaborative filtering assumes that the business has obtained quite complete data from users' ratings of products.

All these variables are visualized in the following graphs.

- Item category

![category](https://github.com/alvinrach/Product-Recommender-System/blob/main/category.png?raw=true)

- Gender specificity of a product

![item_gender_model_attr](https://github.com/alvinrach/Product-Recommender-System/blob/main/item_gender_model_attr.png?raw=true)

- Rating distribution

![rating](https://github.com/alvinrach/Product-Recommender-System/blob/main/rating.png?raw=true)

- Product release year

![release_year](https://github.com/alvinrach/Product-Recommender-System/blob/main/release_year.png?raw=true)

In addition, in order to understand the data, it is also done:
1. Check for duplication of data. No duplicate samples
2. Check for null values and data types, using pd.DataFrame.info(). There are null values on feature brand and user_attr

These two things are done so that:
1. There is no indicated duplicate data entered into the dataset
2. There are no null values involved in modeling because it will cause running errors or worsen performance

## Data Preparation
Techniques used to perform data preparation include:
1. Choose the features to be processed according to the type of recommendation system to be used, as previously mentioned
2. Fill values for null values, as in feature brand using fillna

For content-based filtering, additional data preparation is carried out in the form of:
3. Using pd.get_dummies() to do one-hot encoding, where category data will be created as a feature and marked 1 if it exists, and 0 if it doesn't exist
4. Perform pd.DataFrame().groupby() or group-by on item_id with a note that the combined numbers are averaged
5. Perform cosine similarity. Cosine similarity assumes and determines the extent to which two vectors point in the same direction

Meanwhile, in collaborative filtering, additional data preparation is carried out in the form of:
3. Divide the dataset by 80% training and 20% test using train_test_split
4. Scaling/normalization (min-max scaling) on training rating and validation, but only fit using training data

These techniques are carried out so that:
1. Reduce modeling time by selecting relevant features
2. One-hot encoding is done instead of TfidfVectorizer/CountVectorizer if the data obtained is in the form of a review in the form of sentences. But both have the same goal, namely to convert non-numeric data into data that can be calculated on the model
3. Group-by is done in order to get the characteristics of each restaurant. At this point, due to using the mean, the hallmark of a successful group-by is that there are only dummy numbers when checking for unique values.
4. Cosine similarity aims to calculate the extent to which an item is similar to other items, so that the most similar item can be sorted in content-based filtering
5. Split the dataset to test the performance of the model on the test dataset
6. Scaling/normalization (min-max scaling) is done to improve model performance and speed up convergence. Done after sharing the dataset to prevent data leakage

## Modeling
### Content-Based Filtering
After obtaining similarity between items, items can be sorted from those with the highest similarity. The sort does not include the items themselves (because cosine_similarity also measures the similarity of the same item). For example, here are 10 items that are most similar to item 2360.

![content_based_2360](https://github.com/alvinrach/Product-Recommender-System/blob/main/content_based_2360.png?raw=true)

The advantage of this system is that it can be used for new customers. This is because new customer data usually does not have rating data. Therefore content-based filtering can be run with very little data that has been entered in the sample. While the drawback of this measurement is that it is not easy to carry out metric measurements.

### Collaborative Filtering
Specific data for the aforementioned collaborative filtering will then be modeled. First, we will calculate the match between the user and the item using the embedding technique. Then do the dot multiplication between the embedding item and the user. In addition, the addition of bias for each item and user is also done. Finally, a sigmoid activation function is added, thus outputting data in the range 0 to 1, such as the input to the label target.

The experiment is designed using 100 epochs, using EarlyStopping and ModelCheckpoint from the Keras library, so that it can save the best epochs and stop when in a given epoch there is no performance improvement.

After running the model is complete, select a user. Make predictions on each item that has never been purchased by the user. After that, sort the prediction rating and perform the reverse transformation of min-max scaling on the existing prediction rating. For example, the following shows 10 recommendations for user 49390.

![collaborative_user_49390](https://github.com/alvinrach/Product-Recommender-System/blob/main/collaborative_user_49390.png?raw=true)

The advantage of collaborative filtering is that it is not difficult to measure metrics. While the drawback is that it cannot be used if there is no user rating/rating data for the item.

## Evaluation
### Content-Based Filtering
The content-based filtering evaluation uses [precision@k](https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54) where the precision of only k recommendations is measured which are issued. Precision@k is formulated with:

```
precision@k = (# of the relevant @k recommended items) / (# @k recommended items)
```

The definition of a relevant item is an item that has a cosine_similarity value greater than 0.7. As an example of its application, item 2360 has a precision@k of 100% because all values are above the threshold. Here's an implementation of the code.

```
item_id = pd.Series(f.index).sample(1).iloc[0]

def recommendations(item_id=item_id, threshold=0.7, n_item=10):
  print(f'Showing similar item for item {item_id}')
  print()
  a = np.argsort(cosine_sim_df[item_id].values)
  a = a[~np.isin(a,item_id)]
  a = a[-n_item:][::-1]
  a = {i:cosine_sim_df[item_id][i] for i in a}
  a

  b=0
  for i,j in a.items():
    if j>threshold:
      b+=1
    print('Item', i, '|', 'Cosine Similarity Value :', j)
  print()

  c = b*100/n_item
  print('Precision(%):')
  return c
  
recommendations(item_id)
```

![content_based_2360_2](https://github.com/alvinrach/Product-Recommender-System/blob/main/content_based_2360_2.png?raw=true)

In addition, repeated measurements were also carried out with 100 arbitrary items, with one output item as many as 10 pieces with the same threshold (0.7). The measurement results are then averaged. Measurements show the system has a precision@k of 99.8%. Here's an implementation of the code.

```
def _evalRecommendations(item_id=item_id, threshold=0.7, n_item=10):
  a = np.argsort(cosine_sim_df[item_id].values)
  a = a[~np.isin(a,item_id)]
  a = a[-n_item:][::-1]
  a = {i:cosine_sim_df[item_id][i] for i in a}
  a

  b=0
  for i,j in a.items():
    if j>threshold:
      b+=1

  c = b*100/n_item
  return c

def evalRecommendations(n_to_eval=100):
  a = 0
  for i in range(n_to_eval):
    item_id = pd.Series(f.index).sample(1).iloc[0]
    a = a + _evalRecommendations(item_id)

  print(f'Evaluation (Precision) for {n_to_eval} times try is : {a/n_to_eval} %')

evalRecommendations(100)
```

![eval_avg_content_based](https://github.com/alvinrach/Product-Recommender-System/blob/main/eval_avg_content_based.png?raw=true)

The advantage of the precision@k metric is that it is practical, there is no need to compute the entire dataset. The disadvantage of precision@k is that it is sometimes inconsistent / varied, because it only counts on a number of k recommendations issued. However, this variation can be reduced by taking the average as mentioned above.

### Collaborative Filtering
To evaluate, the metric used is Root Mean Squared Error.
- Root Mean Squared Error (RMSE) is one of the metrics used in the case of regression, in addition to Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE). The formula for this metric is formulated as follows:
![RMSE Formula](https://miro.medium.com/max/412/1*RSYTYpqyGDYWPmI0rD8zqA.png)
Where n is the number of samples, y is the actual value, and is the predicted value.
- The advantage of RMSE over other regression metrics is that it is suitable if you want to take into account the error value of a larger outlier, if that is so desired. The disadvantages of RMSE compared to other regression metrics are that it is less easy to interpret (compared to MAE) and makes the metric too sensitive in calculating the error value of outliers.
- Deployment is done by typing tf.keras.metrics.RootMeanSquaredError() in the Tensorflow model metrics when compiling:
```
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate=0.0005),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
```
In this project, RMSE for training was achieved as low as 0.3344 and RMSE validation as low as 0.3634 in the 99th epoch. Even though the epoch was set to 100 epochs, the lowest RMSE was achieved at the 99th epoch, and the ModelChekpoint technique managed to record the model at that epoch. It is possible that RMSE will decrease in the next epoch, but considering the long running time (~20 seconds per epoch) and using the limited GPU provided by Google Colab, it was decided to use only 100 epochs. The following is a plot of model performance per epoch.

![plot_rmse_recommender](https://github.com/alvinrach/Product-Recommender-System/blob/main/plot_rmse_recommender.png?raw=true)



---


# # Product Recommendation System

## Project Overview
Sistem rekomendasi merupakan salah satu bentuk aplikasi pembelajaran mesin. Sistem rekomendasi mempunyai banyak jenis, salah satunya adalah sistem rekomendasi produk. Salah satu bisnis yang menggunakan sistem rekomendasi produk yaitu sektor e-commerce. Rekomendasi produk pada e-commere sendiri penting dikarenakan dapat meningkatkan performa bisnis dari e-commerce.

[Sistem rekomendasi pada e-commerce](https://www.ismll.uni-hildesheim.de/pub/pdfs/Hauger_Tso_Schmidt-Thieme_GFKL2007.pdf) dilakukan untuk memberikan rekomendasi produk yang tepat ke pelanggan untuk kondisi-kondisi tertentu. Contohnya pada Shopee yang memberikan rekomendasi setelah pelanggan membeli barang tertentu.

[Beberapa teknik](https://d1wqtxts1xzle7.cloudfront.net/59762468/10.1.1.695.642820190617-91457-z4s1rf-with-cover-page-v2.pdf?Expires=1636075790&Signature=GHF61q6wBl4xM2--demA-QMVbt8M1m~xlwaxuWZzvZucb7Fm51fn8zxmlu-vmtY0onvuECBIOkJzdpQqOFXl6kd5DNFQpcBGU5IGkvweabciuSfGph7esQY7ppLZnEKDqZGJMes5Oxrx6FCLBVTM3MSC5bBZB3~WStLKxDBA40l1hwfnpFAm1bpilZTWams3HVVHnPW~5LFXipg2MAYlTMQAjmOQBfSCChyezzmzr2kV79qs8jMosE5s-zWR5acQxn3Z7Ey4M5orybysI3BbVUK~RfZJq6hNy7a1ZHFgJKsTOnxy5VsaU5mBLEDHFouyiacmxg0Tsmgzr1bCVfRvww__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) yang digunakan pada sistem rekomendasi antara lain adalah *content-based filtering* dan *collaborative fitering*. Terdapat pula *hybrid recommendation system*, yaitu menggunakan kombinasi dari beberapa sistem *filtering* rekomendasi. Pada *project* ini akan digunakan *content-based filtering* dan *collaborative fitering* untuk membangun sistem rekomendasi produk.

## Business Understanding
### Problem Statements
1. Bagaimana mengeluarkan rekomendasi produk dari dataset tertentu?
2. Bagaimana melakukan pengukuran metrik pada sistem rekomendasi produk?

### Goals
1. Mengeluarkan rekomendasi produk menggunakan *content-based filtering* dan *collaborative filtering*
2. Melakukan pengukuran menggunakan metrik tertentu pada sistem rekomendasi produk

### Solution Approach
#### Content-Based Filtering
*Content-based filtering* merekomendasikan item yang mirip dengan item lainnya pada masa lalu. Sebagai contoh, jika seseorang menyukai boneka, maka sistem merekomendasikan pula yang mempunyai nilai fitur mirip dengan boneka. Boneka dikategorikan sebagai barang wanita, sehingga akan direkomendasikan pula barang wanita ke user yang membeli boneka. Yang baru saja disebutkan hanyalah contoh, dapat pula melibatkan fitur-fitur lainnya untuk membuat sistem rekomendasi tsb.

Kelebihan sistem ini yaitu dapat digunakan untuk pelanggan baru. Ini dikarenakan data pelanggan baru biasanya tidak memiliki data rating. Oleh karena itu *content-based filtering* dapat dijalankan dengan sedikit saja data yang sudah masuk pada sampel tsb. Sedangkan kekurangan pengukuran ini yaitu, tidak mudah untuk dilakukan pengukuran metrik.

Prinsip dari sistem ini ialah mengukur skor kesamaan setiap item dengan setiap item lainnya, dan selanjutnya dilakukan pengurutan skor kesamaan. Beberapa item yang memiliki skor kesamaan paling tinggi akan ditampilkan sebagai hasil rekomendasi. 

Untuk pengolahan input, jika terdapat review berupa kalimat maka digunakan TfIdfVectorizer atau CountVectorizer. Dan jika terdapat fitur kategorikal, digunakan one-hot encoding. Selanjutnya pengukuran kesamaan menggunakan beberapa algoritma, salah satunya menggunakan cosine_similarity.

#### Collaborative Filtering
*Collaborative filtering* merekomendasikan item berdasarkan penilaian dari user terhadap item. Sistem ini terbagi menjadi *memory-based* dan *model-based*. *Memory-based* sendiri terbagi lagi menjadi user-based dan item-based. Sedangkan *model-based* terbagi menjadi *cluster-based*, *matrix factorization* dan *deep learning*.

*User-based* mencari pengguna dengan selera sama, misalnya sama-sama menyukai film komedi. *Item-based* menghitung kesamaan antara masing-masing item dari semua user. *Cluster-based* menggunakan algoritma *clustering* (K-Means, Gaussian Mixture Model, DBScan) untuk mengelompokkan user sesuai variabel yang tersedia. *Matrix Factorization* menggunakan pembentukan dekomposisi matriks. Sedangkan sistem dengan deep learning menggunakan algortima *deep learning* untuk menentukan rekomendasi akhir.

Pada pendekatan sistem rekomendasi produk secara *collaborative filtering* dengan menggunakan teknik embedding kali ini akan digunakan data user, item dan rating user terhadap item tsb. Setiap user dan item akan dilakukan embedding untuk dimodelkan dengan target yaitu rating itu sendiri.

Kelebihan *collaborative filtering* adalah tidak sulit untuk dilakukan pengukuran metrik. Sedangkan kekurangannya adalah tidak bisa digunakan jika belum terdapat data penilaian/rating user terhadap item.

## Data Understanding
[Sumber data](https://www.kaggle.com/ruchi798/marketing-bias-in-product-recommendations) yang digunakan berasal dari Kaggle. Pada *source code*, data ditarik dengan menggunakan [API Kaggle](https://www.kaggle.com/docs/api#interacting-with-datasets). Untuk mengambil data secara manual dilakukan dengan mengunduh langsung dari link sumber data tsb. Dataset yang digunakan memiliki 1292954 sampel/rows non duplikasi.

Ada beberapa variabel/feature pada dataset tsb, namun yang akan digunakan pada proyek kali ini ialah:
- item_id : Id produk. Dikarenakan pada dataset tidak memiliki nama produk, akan digunakan item_id sebagai penunjuk nama produk. Terdapat 9560 jenis produk.
- user_id : Id user.  Terdapat 1157633 user unik.
- rating : Rating yang diberikan pelanggan saat pesanan selesai.
- model_attr : Model produk, apakah produk tsb ditujukan untuk wanita, pria, ataupun keduanya.
- category : Kategori dari produk. Terdapat 10 jenis kategori.  
- brand : Brand dari produk. Beberapa produk tidak memiliki brand, sehingga selanjutnya akan dilabel unbranded menggunakan fillna.
- year : Tahun pembuatan produk.

Yang akan digunakan pada *content-based filtering* ialah item_id, category, brand dan year. Rating tidak digunakan karena mengasumsikan *content-based filtering* kali ini ditujukan untuk bisnis yang belum mengumpulkan dataset yang cukup lengkap rating-nya, sehingga digunakan data selain rating.

Sedangkan yang akan digunakan untuk *collaborative filtering* adalah item_id, user_id dan rating. *Collaborative filtering* mengasumsikan bahwa bisnis telah mendapatkan data yang cukup lengkap dari rating para pengguna terhadap produk-produk.

Serba-serbi variabel tsb divisualisasikan dalam grafik-grafik berikut ini.

- Kategori barang

![category](https://github.com/alvinrach/Product-Recommender-System/blob/main/category.png?raw=true)

- Pengkhususan gender suatu produk

![item_gender_model_attr](https://github.com/alvinrach/Product-Recommender-System/blob/main/item_gender_model_attr.png?raw=true)

- Distribusi rating

![rating](https://github.com/alvinrach/Product-Recommender-System/blob/main/rating.png?raw=true)

- Tahun rilis produk

![release_year](https://github.com/alvinrach/Product-Recommender-System/blob/main/release_year.png?raw=true)

Selain itu, untuk memahami data, dilakukan pula:
1. Mengecek data duplikasi. Tidak terdapat sampel terduplikasi
2. Mengecek _null values_ dan jenis data, dengan menggunakan pd.DataFrame.info(). Terdapat _null values_ pada feature brand dan user_attr

Kedua hal ini dilakukan agar:
1. Tidak ada data yang terindikasi duplikat masuk ke dalam dataset
2. Tidak ada _null values_ yang ikut dalam *modelling* karena akan menyebabkan *running error* atau memperburuk performa

## Data Preparation 
Teknik yang digunakan untuk melakukan preparasi data diantaranya yaitu:
1. Memilih fitur yang akan diolah menurut tipe sistem rekomendasi yang akan digunakan, seperti yang telah disebutkan sebelumnya
2. Mengisi nilai untuk _null values_, seperti pada *feature* brand menggunakan fillna

Untuk *content-based filtering*, dilakukan tambahan preparasi data berupa:
3. Menggunakan pd.get_dummies() untuk melakukan *one-hot encoding*, dimana data kategori akan dibuat sebagai feature dan ditandai 1 jika ada, dan 0 jika tidak ada
4. Melakukan pd.DataFrame().groupby() atau group-by terhadap item_id dengan catatan angka yang digabungkan dilakukan perata-rataan
5. Melakukan cosine similarity. Cosine similarity mengasumsikan dan menentukan sejauh mana dua vektor menunjuk ke arah yang sama

Sedangkan pada *collaborative filtering*, dilakukan tambahan preparasi data berupa:
3. Membagi dataset sebanyak 80% latih dan 20% test menggunakan train_test_split
4. _Scaling_/normalisasi (min-max scaling) pada rating latih dan validasi, namun hanya melakukan fit menggunakan data latih

Teknik-teknik tsb dilakukan agar:
1. Mengurangi waktu modelling dengan memilih fitur relevan
2. One-hot encoding dilakukan sebagai pengganti TfidfVectorizer/CountVectorizer jika data yang didapat berupa review berupa kalimat. Namun keduanya bertujuan sama, yaitu untuk mengubah data non-numerik menjadi data yang dapat dikalkulasi pada model
3. Group-by dilakukan agar didapatkan karakteristik dari setiap restoran. Pada titik ini, dikarenakan menggunakan mean, ciri-ciri group-by yang berhasil adalah hanya terdapat angka dummy saat dilakukan pengecekan nilai unik.
4. Cosine similarity bertujuan untuk mengkalkulasikan sejauh apa kemiripan sebuah item dengan item lainnya, sehingga dapat dilakukan penyortiran item yang paling mirip pada *content-based filtering*
5. Membagi dataset untuk menguji performa model pada tes dataset
6. _Scaling_/normalisasi (min-max scaling) dilakukan untuk meningkatkan performa model dan mempercepat konvergensi. Dilakukan setelah pembagian dataset untuh mencegah *data leakage*

## Modeling
### Content-Based Filtering
Setelah didapatkan similarity antar item, item dapat disortir dari yang memiliki similarity paling tinggi. Sortiran tidak termasuk item itu sendiri (dikarenakan cosine_similarity juga mengukur similarity item yang sama). Sebagai contoh, berikut merupakan 10 item yang paling mirip dengan item 2360.

![content_based_2360](https://github.com/alvinrach/Product-Recommender-System/blob/main/content_based_2360.png?raw=true)

Kelebihan sistem ini yaitu dapat digunakan untuk pelanggan baru. Ini dikarenakan data pelanggan baru biasanya tidak memiliki data rating. Oleh karena itu *content-based filtering* dapat dijalankan dengan sedikit saja data yang sudah masuk pada sampel tsb. Sedangkan kekurangan pengukuran ini yaitu, tidak mudah untuk dilakukan pengukuran metrik.

### Collaborative Filtering
Data khusus untuk *collaborative filtering* yang telah disebutkan sebelumnya, selanjutnya akan dimodelkan. Pertama-tama akan dilakukan perhitungan kecocokan antara user dan item dengan menggunakan teknik embedding. Selanjutnya dilakukan perkalian dot antara embedding item dan user. Selain itu dilakukan pula penambahan bias untuk setiap item dan user. Terakhir, ditambahkan fungsi aktivasi sigmoid, sehingga mengeluarkan data dalam rentang 0 hingga 1, seperti input pada target label.

Percobaan dirancang dengan menggunakan 100 epoch, menggunakan EarlyStopping dan ModelCheckpoint dari library Keras, sehingga dapat menyimpan epoch terbaik dan berhenti ketika dalam epoch tertentu tidak terdapat peningkatan performa.

Setelah running model selesai, pilih satu user. Lakukan prediksi pada setiap item yang belum pernah dibeli user tsb. Setelah itu, sortir rating prediksi dan lakukan transformasi kebalikan dari min-max scaling pada rating prediksi yang telah ada. Sebagai contoh, berikut ditampilkan 10 rekomendasi untuk user 49390.

![collaborative_user_49390](https://github.com/alvinrach/Product-Recommender-System/blob/main/collaborative_user_49390.png?raw=true)

Kelebihan *collaborative filtering* adalah tidak sulit untuk dilakukan pengukuran metrik. Sedangkan kekurangannya adalah tidak bisa digunakan jika belum terdapat data penilaian/rating user terhadap item.

## Evaluation
### Content-Based Filtering
Pengevaluasian *content-based filtering* menggunakan [precision@k](https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54) dimana hanya diukur presisi sejumlah k rekomendasi yang dikeluarkan. Precision@k dirumuskan dengan:

```
precision@k = (# item terekomendasi @k yang relevan) / (# item terekomendasi @k)
```

Definisi item yang relevan adalah item yang memiliki cosine_similarity value lebih besar dari 0.7. Sebagai contoh penerapannya, untuk item 2360 mempunyai precision@k sebesar 100% dikarenakan semua value diatas threshold. Berikut merupakan penerapan pada kode.

```
item_id = pd.Series(f.index).sample(1).iloc[0]

def recommendations(item_id=item_id, threshold=0.7, n_item=10):
  print(f'Showing similar item for item {item_id}')
  print()
  a = np.argsort(cosine_sim_df[item_id].values)
  a = a[~np.isin(a,item_id)]
  a = a[-n_item:][::-1]
  a = {i:cosine_sim_df[item_id][i] for i in a}
  a

  b=0
  for i,j in a.items():
    if j>threshold:
      b+=1
    print('Item', i, '|', 'Cosine Similarity Value :', j)
  print()

  c = b*100/n_item
  print('Precision (%):')
  return c
  
recommendations(item_id)
```

![content_based_2360_2](https://github.com/alvinrach/Product-Recommender-System/blob/main/content_based_2360_2.png?raw=true)

Selain itu, dilakukan pula pengukuran berulang dengan 100 item sembarang, dengan satu item keluaran sebanyak 10 buah dengan threshold yang sama (0.7). Hasil pengukuran kemudian dirata-ratakan. Pengukuran menunjukkan sistem memiliki precision@k sebesar 99.8%. Berikut merupakan penerapan pada kode.

```
def _evalRecommendations(item_id=item_id, threshold=0.7, n_item=10):
  a = np.argsort(cosine_sim_df[item_id].values)
  a = a[~np.isin(a,item_id)]
  a = a[-n_item:][::-1]
  a = {i:cosine_sim_df[item_id][i] for i in a}
  a

  b=0
  for i,j in a.items():
    if j>threshold:
      b+=1

  c = b*100/n_item
  return c

def evalRecommendations(n_to_eval=100):
  a = 0
  for i in range(n_to_eval):
    item_id = pd.Series(f.index).sample(1).iloc[0]
    a = a + _evalRecommendations(item_id)

  print(f'Evaluation (Precision) for {n_to_eval} times try is : {a/n_to_eval} %')

evalRecommendations(100)
```

![eval_avg_content_based](https://github.com/alvinrach/Product-Recommender-System/blob/main/eval_avg_content_based.png?raw=true)

Kelebihan metrik precision@k adalah praktis, tidak perlu menghitung secara keseluruhan dataset. Kekurangan precision@k adalah terkadang tidak konsisten/variatif, karena hanya menghitung pada sejumlah k rekomendasi yang dikeluarkan. Namun begitu, variasi tsb dapat dikurangi dengan mengambil rata-rata seperti yang disebutkan diatas.

### Collaborative Filtering
Untuk mengevaluasi, metrik yang digunakan ialah _Root Mean Squared Error_.
- _Root Mean Squared Error_ (RMSE) adalah salah satu metrik yang digunakan pada kasus regresi, disamping _Mean Absolute Error_ (MAE) dan _Mean Absolute Percentage Error_ (MAPE). Formula dari metrik ini dirumuskan sebagai berikut:
![RMSE Formula](https://miro.medium.com/max/412/1*RSYTYpqyGDYWPmI0rD8zqA.png)
Dimana n merupakan jumlah sampel, y adalah nilai aktual, dan ŷ merupakan nilai prediksi.
- Kelebihan RMSE dibanding metrik regresi lainnya ialah cocok jika ingin memperhitungkan nilai eror dari pencilan lebih besar, jika memang dikehendaki demikian. Kekurangan RMSE dibanding metrik regresi lainnya yaitu lebih tidak mudah untuk diinterpretasikan (dibanding MAE) dan membuat metrik terlalu sensitif dalam memperhitungkan nilai eror dari pencilan.
- Penerapan dilakukan dengan cara mengetik tf.keras.metrics.RootMeanSquaredError() pada metrik model Tensorflow saat mengompilasi:
``` 
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate=0.0005),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
```
Pada project ini, dicapai RMSE untuk training serendah 0.3344 dan RMSE validation serendah 0.3634 pada epoch ke 99. Meski epoch di-set sebesar 100 epoch, namun RMSE terendah tercapai pada epoch ke 99, dan teknik ModelChekpoint berhasil merekam model pada epoch tersebut. Ada kemungkinan RMSE berkurang pada epoch selanjutnya, namun mempertimbangkan waktu *running* yang panjang (~20 detik per epoch) dan menggunakan GPU yang disediakan Google Colab yang terbatas, diputuskan hanya menggunakan 100 epoch. Berikut merupakan plot performa model per epoch.

![plot_rmse_recommender](https://github.com/alvinrach/Product-Recommender-System/blob/main/plot_rmse_recommender.png?raw=true)

**---This is the end of the markdown---**
