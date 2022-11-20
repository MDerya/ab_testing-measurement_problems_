

#       AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması

"""
    İş Problemi

    Facebook kısa süre önce mevcut "maximum bidding" adı verilen teklif verme türüne alternatif olarak
    yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
    bu yeni özelliği test etmeye karar verdi ve average bidding'in maximum bidding'den daha fazla
    dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor
    ve bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.
    Bombabomba.comiçin nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için
    Purchase metriğine odaklanılmalıdır.


    Veri Seti Hikayesi

    Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
    reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.
    Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleri ab_testing.xlsx
    excel’inin ayrı sayfalarında yer almaktadır. Kontrol grubuna Maximum Bidding,
    test grubuna Average Bidding uygulanmıştır.

    Impression : Reklam görüntüleme sayısı
    Click      : Görüntülenen reklama tıklama sayısı
    Purchase   :Tıklanan reklamlar sonrası satın alınan ürün sayısı
    Earning    :Satın alınan ürünler sonrası elde edilen kazanç

"""

############################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
############################################
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Adım 1:  ab_testing.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz.
# Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.


df_c = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
df_t = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

df_c.describe().T

df_t.describe().T

df_c.info()
df_t.info()

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.


"""df_ct = pd.concat([df_c, df_t])

df_ct.tail() #indexte bir sorun var

df_ct= df_ct.reset_index() #indexteki sorun çözülmedi"""

df_ct = pd.concat([df_c, df_t], ignore_index = True)

############################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
############################################


#Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2 average bidding ile maximum bidding'e ait purchase ortalamaları arasında ist ol.anl.fark yoktur
# H1 : M1!= M2 ..... vardır.



# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz.

df_c["Purchase"].mean() #550
df_t["Purchase"].mean() #582


############################################
# Görev 3:  Hipotez Testinin Gerçekleştirilmesi
############################################

#Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir. Kontrol ve test grubunun
# normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz.

# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu ? Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = shapiro(df_c["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df_t["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

""" p-value = 0.5891, p-value = 0.1541 
    her iki p_value degeri de >0.05 oldugundan H0 reddedilemez.
    Yani normal dagılım saglanmaktadır."""

# VaryansHomojenliği :
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p < 0.05 H0 RED , p > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre varyans homojenligi varsayımı sağlanıyor mu? Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = levene(df_c["Purchase"], df_t["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

""" p-value = 0.1083 > 0.05 oldugundan H0 reddedilemez.
    Yani varyanslar homojendir."""



# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.

"""  Varsayımlar sağlanıyor, bağımsız iki örneklem t testi (parametrik test):
     ttest metodu der ki;
    -eger normallik varsayımı saglanıyorsa beni kullanabilirsin,
    -normallik saglanıyor ve varyans homojenligi saglanıyorsa da beni kullanabilirsin,
    -normallik saglanıyor ,varyans homojenligi saglanmıyorsa da beni kullanabilirsin(bu durumda 'equal_var=False' gir)
"""

test_stat, pvalue = ttest_ind(df_c["Purchase"], df_t["Purchase"], equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))




# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu
# satın alma ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız


""" p-value = 0.3493 > 0.05 oldugundan H0 reddedilemez.
    Yani;
    average bidding ile maximum bidding'e ait purchase ortalamaları arasında ist ol.anl.fark yoktur.
"""


############################################
# Görev 4:  Sonuçların Analizi
############################################


#Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

"""  Varsayımlar sağlanıyor, bağımsız iki örneklem t testi (parametrik test):
     ttest metodu der ki;
    -eger normallik varsayımı saglanıyorsa beni kullanabilirsin,
    -normallik saglanıyor ve varyans homojenligi saglanıyorsa da beni kullanabilirsin,
    -normallik saglanıyor ,varyans homojenligi saglanmıyorsa da beni kullanabilirsin(bu durumda 'equal_var=False' gir)
"""

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.


"""
    df_c["Purchase"].mean() #550
    df_t["Purchase"].mean() #582
    Grupların ayrı ayrı ortalamaları alındıgında sanki aralarında bir fark varmış gibi duruyor 
    ama istatistiki olarak anlamlı bir fark yoktur,bu fark şans eseri çıkmış olabilir.
    Dolayısıyla yeni teklif türünü size ek bir maliyet getirecekse kullanmayabilirsiniz.
    
"""