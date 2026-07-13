from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, trim, avg, max as _max, min as _min, countDistinct, lower, regexp_replace, when, round, split, concat_ws
from pyspark.sql.window import Window
from pyspark.sql.functions import rank
import pyspark
import os
import time

print(f"PySpark version: {pyspark.__version__}")

# =============================
# 1- Créer une session Spark avec MongoDB
# =============================
jars_path = os.path.join(os.getcwd(), "jars")
jar_files = [
    os.path.join(jars_path, "mongo-spark-connector_2.12-10.3.0.jar"),
    os.path.join(jars_path, "mongodb-driver-sync-5.1.0.jar"),
    os.path.join(jars_path, "mongodb-driver-core-5.1.0.jar"),
    os.path.join(jars_path, "bson-5.1.0.jar"),
    os.path.join(jars_path, "bson-record-codec-5.1.0.jar")
]

spark = SparkSession.builder \
    .appName("EcommerceAnalysis") \
    .master("local[*]") \
    .config("spark.jars", ",".join(jar_files)) \
    .config("spark.mongodb.write.connection.uri", "mongodb+srv://mayssalabiadh34_db_user:hNVmc3UnTNokzTmP@ecommercecluster.samm9jn.mongodb.net/EcommerceDB") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# =============================
# 2- Lire les 3 datasets des sources différentes 
# =============================
df_local = spark.read.csv("sales_part_0.csv", header=True, inferSchema=True) #Data local 
df_vm = spark.read.csv(r"C:\Users\Labiadh Mayssa\Documents\sharedfolder\sales_part_2.csv", header=True, inferSchema=True) #Data du Machine Virtuelle 
df_gdrive = spark.read.csv(r"C:\Users\Labiadh Mayssa\Documents\GDriveDataset\sales_part_1.csv", header=True, inferSchema=True) #Data du Google Drive

# =============================
# 3-  Nettoyage des données 
# =============================
def clean_df_enriched(df):
    return (
        df.dropna(subset=["CustomerID", "Quantity", "UnitPrice", "Country"])
        .filter((col("Quantity") > 0) & (col("UnitPrice") > 0))
        .dropDuplicates()
        .withColumn("Description", trim(lower(regexp_replace(col("Description"), "[^a-zA-Z0-9 ]", ""))))
        .withColumn("Revenue", col("Quantity") * col("UnitPrice"))
        .withColumn("BigOrder", when(col("Revenue") > 500, 1).otherwise(0))
    )

#Nettoyage de tous les data
df_local_clean = clean_df_enriched(df_local)
df_vm_clean = clean_df_enriched(df_vm)
df_gdrive_clean = clean_df_enriched(df_gdrive)
#Répartition des data en 4
df_local_clean = df_local_clean.repartition(4)
df_vm_clean = df_vm_clean.repartition(4)
df_gdrive_clean = df_gdrive_clean.repartition(4)

# =============================
# 4- Combiner les datasets
# =============================
df_all = df_local_clean.unionByName(df_vm_clean).unionByName(df_gdrive_clean)
df_all.cache()
print(f"Combinaison terminée: {df_all.count()} lignes")
# ===============================================================
# 5- Fonctions de Transformations et d'Action Spark / Agrégations
# ===============================================================

# 5a. Ventes par produit
df_sales_product = (
    df_all.groupBy("StockCode", "Description")
    .agg(
        round(_sum("Revenue"), 2).alias("TotalSales"),
        round(_max("Revenue"), 2).alias("MaxSale"),
        round(_min("Revenue"), 2).alias("MinSale"),
        round(avg("Revenue"), 2).alias("AvgSale"),
        countDistinct("CustomerID").alias("NumCustomers")
    )
    .orderBy(col("TotalSales").desc())
)
print("1. Affichage des Ventes par produit \n")
df_sales_product.show(10)

# 5b. Ventes par pays
df_sales_country = (
    df_all.groupBy("Country")
    .agg(
        round(_sum("Revenue"), 2).alias("TotalRevenue"),
        round(avg("Revenue"), 2).alias("AvgRevenue"),
        _sum("BigOrder").cast("long").alias("BigOrdersCount")
    )
    .orderBy(col("TotalRevenue").desc())
)
print("2. Affichage des Ventes par pays \n")
df_sales_country.show(10)

# 5c. Ventes par produit et pays
df_sales_product_country = (
    df_all.groupBy("Country", "StockCode", "Description")
    .agg(
        round(_sum("Revenue"), 2).alias("TotalSales"),
        countDistinct("CustomerID").alias("NumCustomers"),
        _sum("BigOrder").cast("long").alias("BigOrdersCount")
    )
    .orderBy(col("Country"), col("TotalSales").desc())
)
print("3. Affichage des Ventes par produit et pays \n")
df_sales_product_country.show(10)

# 5d. Top 5 produits par pays avec window
windowSpec = Window.partitionBy("Country").orderBy(col("TotalSales").desc())
df_top_products_country = (
    df_sales_product_country
    .withColumn("Rank", rank().over(windowSpec))
    .filter(col("Rank") <= 5)
)
print("4. Affichage des Top 5 produits par pays avec Window \n")
df_top_products_country.show(5)

# 5e. Ventes par mois
df_all = df_all.withColumn("InvoiceDateClean", regexp_replace("InvoiceDate", ",", "/"))
df_all = df_all.withColumn(
    "YearMonth",
    concat_ws("-",
        split(col("InvoiceDateClean"), "/").getItem(2),
        split(col("InvoiceDateClean"), "/").getItem(1)
    )
)

df_sales_month = (
    df_all.groupBy("YearMonth")
    .agg(
        round(_sum("Revenue"), 2).alias("TotalRevenueMonth"),
        _sum("BigOrder").cast("long").alias("BigOrdersMonth")
    )
    .orderBy("YearMonth")
)
print("5. Affichage des Ventes par mois \n")
df_sales_month.show(10)

# 5f. Produits à marge élevée (Profit)
df_all = df_all.withColumn("Profit", round(col("Revenue") - col("UnitPrice")*0.7, 2))
df_high_profit_products = (
    df_all.groupBy("StockCode", "Description")
    .agg(round(_sum("Profit"), 2).alias("TotalProfit"))
    .orderBy(col("TotalProfit").desc())
)
print("6. Affichage des Produits à marge élevée \n")
df_high_profit_products.show()

# 5g. Distribution des quantités
df_all = df_all.withColumn(
    "QuantityRange",
    when(col("Quantity") < 10, "Small")
    .when((col("Quantity") >= 10) & (col("Quantity") < 50), "Medium")
    .otherwise("Large")
)
df_quantity_range = (
    df_all.groupBy("QuantityRange")
    .agg(
        round(_sum("Revenue"), 2).alias("RevenueByRange"),
        countDistinct("StockCode").alias("NumProducts")
    )
)
print("7. Affichage de la distribution des quantités : \n")
df_quantity_range.show()

# ======================================
# 6- FONCTION DE SAUVEGARDE DANS MONGODB
# ======================================
def save_to_mongo(df_spark, collection_name):
    try:
        # Afficher le schéma pour vérification
        print(f"\n--- Schéma de {collection_name} ---")
        df_spark.printSchema()
        
        # Sauvegarder dans MongoDB
        df_spark.write \
            .format("mongodb") \
            .mode("overwrite") \
            .option("database", "EcommerceDB") \
            .option("collection", collection_name) \
            .save()
        print(f"Collection '{collection_name}' sauvegardée dans MongoDB ({df_spark.count()} documents)")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de {collection_name}: {str(e)}")

# =============================
# 7- Sauvegarde dans MongoDB
# =============================
print("SAUVEGARDE DANS MONGODB :")

save_to_mongo(df_sales_product, "SalesByProduct")
save_to_mongo(df_sales_country, "SalesByCountry")
save_to_mongo(df_sales_product_country, "SalesByProductCountry")
save_to_mongo(df_top_products_country, "TopProductsByCountry")
save_to_mongo(df_sales_month, "SalesByMonth")
save_to_mongo(df_high_profit_products, "HighProfitProducts")
save_to_mongo(df_quantity_range, "QuantityRangeDistribution")

#Garder les collections dans des fichiers .CSV
os.makedirs("powerbi_input", exist_ok=True)
df_sales_product.coalesce(1).write.mode("overwrite").option("header", "true").csv("powerbi_input/SalesByProduct")
df_sales_country.coalesce(1).write.mode("overwrite").option("header", "true").csv("powerbi_input/SalesByCountry")
df_sales_product_country.coalesce(1).write.mode("overwrite").option("header", "true").csv("powerbi_input/SalesByProductCountry")
df_top_products_country.coalesce(1).write.mode("overwrite").option("header", "true").csv("powerbi_input/TopProductsByCountry")
df_sales_month.coalesce(1).write.mode("overwrite").option("header", "true").csv("powerbi_input/SalesByMonth")
df_high_profit_products.coalesce(1).write.mode("overwrite").option("header", "true").csv("powerbi_input/HighProfitProducts")
df_quantity_range.coalesce(1).write.mode("overwrite").option("header", "true").csv("powerbi_input/QuantityRangeDistribution")

# =============================
# 8- Fermer Spark
# =============================
time.sleep(3)
spark.stop()
print("\nScript terminé avec succès")