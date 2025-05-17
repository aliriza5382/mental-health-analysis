# 22100011066
# Ali Rıza ŞAHİN

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import missingno as msno
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def analyze_missing_data(df):
    """
    Eksik verilerin rastgele olup olmadığını analiz eder.
    1. Eksik veri oranlarını ve yüzdesini gösterir.
    2. Kategorik değişkenlerle ilişki için Ki-Kare Testi uygular.
    3. Sayısal değişkenlerle ilişki için t-Testi uygular.
    4. Lojistik regresyon ile eksikliğin tahmini yapılır.
    5. Eksik veri tipi (MCAR, MAR, MNAR) belirlenir.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import chi2_contingency, ttest_ind
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    import numpy as np

    # 1. Eksik veri sayıları ve yüzdeleri
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({"Eksik Değer Sayısı": missing_values, "Yüzde (%)": missing_percent})
    missing_df = missing_df[missing_df["Eksik Değer Sayısı"] > 0]

    # Görselleştirme
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
    plt.title("Eksik Veri Isı Haritası")
    plt.show()

    # 2. Ki-Kare Testi (Kategorik değişkenlerle ilişki)
    chi2_results = []
    for col in df.select_dtypes(include=["object"]):
        if df[col].isnull().sum() == 0:
            continue
        table = pd.crosstab(df[col].isnull(), df[col].dropna())
        if table.shape[1] > 1:
            chi2, p, _, _ = chi2_contingency(table)
            chi2_results.append({"Değişken": col, "Ki-Kare Değeri": chi2, "p-Değeri": p})
    chi2_results_df = pd.DataFrame(chi2_results)

    # 3. t-Test (Sayısal değişkenlerle ilişki)
    ttest_results = []
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        temp = df.copy()
        temp["missing"] = temp[col].isnull().astype(int)
        for num_col in df.select_dtypes(include=[np.number]):
            if num_col != col:
                group1 = temp[temp["missing"] == 1][num_col].dropna()
                group2 = temp[temp["missing"] == 0][num_col].dropna()
                if len(group1) > 0 and len(group2) > 0:
                    stat, p_value = ttest_ind(group1, group2, equal_var=False)
                    ttest_results.append({"Eksik Veri Değişkeni": col, "Sayısal Değişken": num_col, "p-Değeri": p_value})
    ttest_results_df = pd.DataFrame(ttest_results)

    # 4. Lojistik Regresyon (Eksikliği tahmin etme)
    logistic_results = []
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        temp = df.copy()
        temp["missing"] = temp[col].isnull().astype(int)
        features = [c for c in df.columns if c != col and df[c].notnull().sum() > 0]
        temp_encoded = temp[features].copy()
        for feat in features:
            if temp_encoded[feat].dtype == "object":
                temp_encoded[feat] = LabelEncoder().fit_transform(temp_encoded[feat].astype(str))
        temp_encoded = temp_encoded.dropna()
        if temp_encoded.shape[0] > 10:
            X = temp_encoded
            y = temp.loc[temp_encoded.index, "missing"]
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            score = model.score(X, y)
            logistic_results.append({"Eksik Veri Değişkeni": col, "Lojistik Regresyon Skoru": score})
    logistic_results_df = pd.DataFrame(logistic_results)

    # 5. Eksik veri türü tahmini (MCAR, MAR, MNAR)
    max_log_score = logistic_results_df["Lojistik Regresyon Skoru"].max() if not logistic_results_df.empty else 0

    if not chi2_results_df.empty and (chi2_results_df["p-Değeri"] < 0.05).any():
        result = "Eksik veriler bazı gözlenen değişkenlerle ilişkili görünüyor, bu yüzden MAR olabilir."
    elif max_log_score >= 0.8:
        result = "Eksik veriler eksik olan değişkenin kendisiyle doğrudan ilişkili olabilir, bu yüzden MNAR olabilir."
    else:
        result = "Eksik veriler tamamen rastgele (MCAR) olabilir."

    return missing_df, chi2_results_df, ttest_results_df, logistic_results_df, result


# Dosyanın yolunu belirleyerek oku
df = pd.read_csv("Mental Health Dataset.csv")

# Eksik verileri görselleştirme:
# 1-) Bar grafiği
msno.bar(df).figure.set_size_inches(10, 5)
plt.show()

# 2-) Matris grafiği
msno.matrix(df).figure.set_size_inches(10, 5)
plt.show()

# Eksik veri analiz fonksiyonu çalıştırılıyor
missing_df, chi2_results_df, ttest_results_df, logistic_results_df, result = analyze_missing_data(df)
print("\n* Eksik Veri Türü Analizi Sonucu:")
print(result)

"""
# *****Sayısal veri olup olmadığı kontrol edildi olsaydı modelin daha iyi işlemesi için
# ayrıklaştırma yapılacaktı.
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\n\n\nSayısal Değişkenler:")
print(numeric_columns, "\n\n\n")

# *****Ayrıklaştırma Örneği:
Şimdi veri kümenizde bulunan bazı sayısal değişkenleri kategorilere ayıralım.

Örnek 1: Yaş Değişkenini Gruplama
Diyelim ki veri kümenizde "Age" (yaş) değişkeni var. Bunu 3 kategoriye bölebiliriz:

Genç (0-25)
Orta Yaşlı (26-50)
Yaşlı (51-100)
Bunun için aşağıdaki kodu kullanabiliriz:


df["Age_Group"] = pd.cut(df["Age"], bins=[0, 25, 50, 100], labels=["Genç", "Orta Yaşlı", "Yaşlı"])
print(df[["Age", "Age_Group"]].head())


# *****Alternatif Olarak bu yapılabilirdi ama ben verilerimin hepsi kategorik olduğu
# için bu yöntemi kullandım: 

1. Kategorik Verileri Kodlayarak Sayısallaştırma
Eğer veri kümen sadece kategorik verilerden oluşuyorsa, bunları sayısal değerlere dönüştürerek işlem yapabiliriz.

Örneğin, "self_employed" değişkeni "Yes", "No", "Unknown" gibi değerler alıyorsa, bunları sayısal hale getirebiliriz:

from sklearn.preprocessing import LabelEncoder

# Tüm kategorik değişkenleri sayısal değerlere çevir
df_encoded = df.apply(LabelEncoder().fit_transform)

print(df_encoded.head())
Bu, tüm kategorik değişkenleri 0, 1, 2, ... gibi sayısal değerlere çevirir ve ardından ayrıklaştırma uygulanabilir.

2. Yeni Sayısal Değişkenler Türetme
Bazen veri kümen gizli sayısal bilgiler içerir ama doğrudan sayısal değildir. Örneğin:

"Days_Indoors" (Evde Kalma Süresi) → Belirli gruplara ayrılabilir (Az, Orta, Çok).
"Work_Interest" (Çalışma İlgisi) → Sayısal hale getirilip analiz edilebilir.
"Social_Weakness" (Sosyal Zayıflık) → Belli aralıklara bölünebilir.
"""


df["self_employed"] = df["self_employed"].fillna("Unknown")

# Üstteki harici eksik veriyle başka başa çıkma yöntemleri:
#   1-) Silme Yöntemi:
#       - Eksik veriye sahip satırları silme (dropna())
#       - Belirli değişkenleri (eksik değeri fazla olan sütunları) silme (drop(columns[]))
#   2-) Eksik Veriyi Doldurma
#       - Eksik değerleri sütunun ortalama (mean()), medyan (median()) veya mod (mode()) değerleri doldurma
#       - Kategorik verilerde en yaygın değerle doldurma
#       - Zamana bağlı verilerde eksik değerleri en sık görülen kategoriyle doldurmak
#   3-) Tahmin Yöntemleri
#       - Regresyon ile tahminleme
#       - Makine öğrenmesi ile eksik değerleri doldurma

# Sayısal değişkenlerin özet istatistikleri
print(df.describe(), "\n")

# Veri setinin genel yapısı
print(df.info(), "\n")

# Eksik verileri kontrol et
print(df.isnull().sum(), "\n")

# Cinsiyet ve ruh sağlığı tedavisi arasındaki ilişki (Ki-Kare Testi)
gender_treatment = pd.crosstab(df["Gender"], df["treatment"])
chi2, p, dof, expected = chi2_contingency(gender_treatment)
print(f"Cinsiyet ve Tedavi - Ki-Kare Değeri: {chi2}, P Değeri: {p}")

# Evde geçirilen gün sayısı ve ailede ruh sağlığı geçmişi
indoor_family = pd.crosstab(df["Days_Indoors"], df["family_history"])
chi2_indoor, p_indoor, dof_indoor, expected_indoor = chi2_contingency(indoor_family)
print(f"Evde Kalma ve Aile Geçmişi - Ki-Kare Değeri: {chi2_indoor}, P Değeri: {p_indoor}")

# Stres seviyesi artışı ve çalışmaya olan ilgi
stress_work = pd.crosstab(df["Growing_Stress"], df["Work_Interest"])
chi2_stress, p_stress, dof_stress, expected_stress = chi2_contingency(stress_work)
print(f"Stres ve Çalışma İlgisi - Ki-Kare Değeri: {chi2_stress}, P Değeri: {p_stress}")

# Aile geçmişi ve tedavi ilişkisi
family_treatment = pd.crosstab(df["family_history"], df["treatment"])
chi2_family_treatment, p_family_treatment, dof_family_treatment, expected_family_treatment = chi2_contingency(
    family_treatment)
print(f"Aile Geçmişi ve Tedavi - Ki-Kare Değeri: {chi2_family_treatment}, P Değeri: {p_family_treatment}")

# Çalışma durumu ve stres seviyesi ilişkisi
work_stress = pd.crosstab(df["self_employed"], df["Growing_Stress"])
chi2_work_stress, p_work_stress, dof_work_stress, expected_work_stress = chi2_contingency(work_stress)
print(f"Çalışma Durumu ve Stres - Ki-Kare Değeri: {chi2_work_stress}, P Değeri: {p_work_stress}")

# Sosyal zayıflık ve tedavi ilişkisi
social_mental_health = pd.crosstab(df["Social_Weakness"], df["treatment"])
chi2_social_mental_health, p_social_mental_health, dof_social_mental_health, expected_social_mental_health = chi2_contingency(
    social_mental_health)
print(f"Sosyal Zayıflık ve Tedavi - Ki-Kare Değeri: {chi2_social_mental_health}, P Değeri: {p_social_mental_health}")

plt.figure(figsize=(10, 5))
sns.heatmap(gender_treatment, annot=True, cmap="Blues", fmt="d")
plt.title("Cinsiyet ve Ruh Sağlığı Tedavi Oranı")
plt.xlabel("Tedavi Görme Durumu")
plt.ylabel("Cinsiyet")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(indoor_family, annot=True, cmap="Greens", fmt="d")
plt.title("Evde Geçirilen Gün Sayısı ve Ailede Ruh Sağlığı Geçmişi")
plt.xlabel("Ailede Ruh Sağlığı Geçmişi")
plt.ylabel("Evde Geçirilen Gün Sayısı")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(stress_work, annot=True, cmap="Oranges", fmt="d")
plt.title("Stres Seviyesi Artışı ve Çalışmaya İlgi")
plt.xlabel("Çalışma İlgisi")
plt.ylabel("Stres Seviyesi Artışı")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(family_treatment, annot=True, cmap="Grays", fmt="d")
plt.title("Aile geçmişi ve tedavi ilişkisi")
plt.xlabel("Tedavi")
plt.ylabel("Aile Geçmişi")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(work_stress, annot=True, cmap="Reds", fmt="d")
plt.title("Çalışma durumu ve stres seviyesi ilişkisi")
plt.xlabel("Stres Seviyesi")
plt.ylabel("Çalışma Durumu")
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(work_stress, annot=True, cmap="Purples", fmt="d")
plt.title("Sosyal zayıflık ve tedavi ilişkisi")
plt.xlabel("Sosyal Zayıflık")
plt.ylabel("Tedavi")
plt.show()

# Kategorik değişkenleri sayısal hale getirme
df_encoded = pd.get_dummies(df, drop_first=True)

# Bağımsız değişkenler (X) ve bağımlı değişken (y)
X = df_encoded.drop(columns=["treatment_Yes"])
y = df_encoded["treatment_Yes"]

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modeli test verisiyle değerlendir
y_pred = model.predict(X_test)


# Farklı modelleri tanımla
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    # "SVM": SVC(kernel="linear"), Çok uzun sürdüğü için yorum satırına alındı
}

# Sonuçları saklamak için
results = {}

# Her bir modeli eğit ve sonuçları yazdır
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n{name} Accuracy Score: {acc:.4f}")
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n{name} Confusion Matrix:\n", cm)

    # Confusion matrix görseli istersen:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# En sonda tüm skorları özetle
print("\n\nModel Performans Karşılaştırması:")
for model_name, acc in results.items():
    print(f"- {model_name}: {acc:.4f}")