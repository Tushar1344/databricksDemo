# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # ETL Optimizations
# MAGIC 
# MAGIC Apache Spark&trade; and Databricks&reg; clusters can be optimized using compression and caching best practices along with autoscaling clusters.
# MAGIC 
# MAGIC ## In this lesson you:
# MAGIC * Compare trade-offs in compression algorithms in terms of parallelization, data transfer, file types, and read vs write time
# MAGIC * Cache data at the optimal point in a workload to limit data transfer across the network
# MAGIC * Configure clusters based on the demands of your workload including size, location, and types of machines
# MAGIC * Employ an autoscaling strategy to dynamically adapt to changes in data size
# MAGIC * Monitor clusters using the Ganglia UI
# MAGIC 
# MAGIC ## Audience
# MAGIC * Primary Audience: Data Engineers
# MAGIC * Additional Audiences: Data Scientists and Data Pipeline Engineers
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC * Web browser: current versions of Google Chrome, Firefox, Safari, Microsoft Edge and 
# MAGIC Internet Explorer 11 on Windows 7, 8, or 10 (see <a href="https://docs.databricks.com/user-guide/supported-browsers.html#supported-browsers#" target="_blank">Supported Web Browsers</a>)
# MAGIC * Concept (optional): <a href="https://academy.databricks.com/collections/frontpage/products/etl-part-1-data-extraction" target="_blank">ETL Part 1 course from Databricks Academy</a>
# MAGIC * Concept (optional): <a href="https://academy.databricks.com/products/etl-part-2-transformations-and-loads-1-user-1-year" target="_blank">ETL Part 2 course from Databricks Academy</a>

# COMMAND ----------

# MAGIC %md
# MAGIC <iframe  
# MAGIC src="//fast.wistia.net/embed/iframe/skaatalsp8?videoFoam=true"
# MAGIC style="border:1px solid #1cb1c2;"
# MAGIC allowtransparency="true" scrolling="no" class="wistia_embed"
# MAGIC name="wistia_embed" allowfullscreen mozallowfullscreen webkitallowfullscreen
# MAGIC oallowfullscreen msallowfullscreen width="640" height="360" ></iframe>
# MAGIC <div>
# MAGIC <a target="_blank" href="https://fast.wistia.net/embed/iframe/skaatalsp8?seo=false">
# MAGIC   <img alt="Opens in new tab" src="https://files.training.databricks.com/static/images/external-link-icon-16x16.png"/>&nbsp;Watch full-screen.</a>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Optimizing ETL Workloads
# MAGIC 
# MAGIC Optimizing Spark jobs boils down to a few common themes, many of which were addressed in other lessons in this course.  Additional optimizations include compression, caching, and hardware choices.
# MAGIC 
# MAGIC Many aspects of computer science are about trade-offs rather than a single, optimal solution.  For instance, data compression is a trade-off between decompression speed, splitability, and compressed to uncompressed data size.  Hardware choice is a trade-off between CPU-bound and IO-bound workloads.  This lesson explores the trade-offs between these issues with rules of thumb to apply them to any Spark workload.
# MAGIC 
# MAGIC While a number of compression algorithms exist, the basic intuition behind them is that they reduce space by reducing redundancies in the data.  Compression matters in big data environments because many jobs are IO bound (or the bottleneck is data transfer) rather than CPU bound.  Compressing data allows for the reduction of data transfer as well as reduction of storage space needed to store that data.
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ETL-Part-3/data-compression.png" style="height: 400px; margin: 20px"/></div>

# COMMAND ----------

# MAGIC %md
# MAGIC Run the cell below to mount the data.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Compression Best Practices
# MAGIC 
# MAGIC There are three compression algorithms commonly used in Spark environments: GZIP, Snappy, and bzip2.  Choosing between this option is a trade-off between the compression ratio, the CPU usage needed to compress and decompress the data, and whether the data it saves is splittable and therefore able to be read and written in parallel.
# MAGIC 
# MAGIC |                   | GZIP   | Snappy | bzip2 |
# MAGIC |:------------------|:-------|:-------|:------|
# MAGIC | Compression ratio | high   | medium | high  |
# MAGIC | CPU usage         | medium | low    | high  |
# MAGIC | Splittable        | no     | yes    | no    |
# MAGIC 
# MAGIC While GZIP offers the highest compression ratio, it is not splittable and takes longer to encode and decode.  GZIP files can only use a single core to process.  Snappy offers less compression but is splittable and quick to encode and decode.  bzip offers a high compression ratio but at high CPU costs, only making it the preferred choice if storage space and/or network transfer is extremely limited.
# MAGIC 
# MAGIC Parquet already uses some compression though Snappy compression applied to Parquet can reduce a file to half its size while improving job performance.  Parquet, unlike other files formats, is splittable regardless of compression format due to the internal layout of the file.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> <a href="https://issues.apache.org/jira/browse/SPARK-14482" target="_blank">The default compression codec for Parquet is Snappy.</a>

# COMMAND ----------

# MAGIC %md
# MAGIC Import a file to test compression options on.

# COMMAND ----------

pagecountsEnAllDF = spark.read.parquet("/mnt/training/wikipedia/pagecounts/staging_parquet_en_only_clean/")

display(pagecountsEnAllDF)    

# COMMAND ----------

# MAGIC %md
# MAGIC Write the file as parquet using no compression, snappy and GZIP

# COMMAND ----------

pathTemplate = username + "/academy/pageCounts{}.csv"
compressions = ["Uncompressed", "Snappy", "GZIP"]

uncompressedPath, snappyPath, GZIPPath = [pathTemplate.format(i) for i in compressions]

pagecountsEnAllDF.write.mode("OVERWRITE").csv(uncompressedPath)
pagecountsEnAllDF.write.mode("OVERWRITE").csv(snappyPath, compression = "snappy")
pagecountsEnAllDF.write.mode("OVERWRITE").csv(GZIPPath, compression = "GZIP")

# COMMAND ----------

# MAGIC %md
# MAGIC Observe the size differences in bytes for the compression techniques

# COMMAND ----------

for compressionType in compressions:
  metadata = dbutils.fs.ls(pathTemplate.format(compressionType))
  size = sum([f.size for f in metadata])
  
  print("{}:  \t{} bytes".format(compressionType, size))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Caching
# MAGIC 
# MAGIC Caching data is one way to improve query performance.  Cached data is maintained on a cluster rather than forgotten at the end of a query.  Without caching, Spark reads the data from its source again after every action. 
# MAGIC 
# MAGIC There are a number of different storage levels with caching, which are variants on memory, disk, or a combination of the too.  By default, Spark's storage level is `MEMORY_AND_DISK`
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> `cache()` is the most common way of caching data while `persist()` allows for setting the storage level.
# MAGIC <img alt="Best Practice" title="Best Practice" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-blue-ribbon.svg"/> It's worth noting that caching should be done with care since data cached at the wrong time can lead to less performant clusters.

# COMMAND ----------

# MAGIC %md
# MAGIC Import a DataFrame to test caching performance trade-offs.

# COMMAND ----------

# MAGIC %python
# MAGIC pagecountsEnAllDF = spark.read.parquet("/mnt/training/wikipedia/pagecounts/staging_parquet_en_only_clean/") 
# MAGIC 
# MAGIC display(pagecountsEnAllDF)    

# COMMAND ----------

# MAGIC %md
# MAGIC Use the `%timeit` function to see the average time for counting an uncached DataFrame.  Recall that Spark will have to reread the data from its source each time

# COMMAND ----------

# MAGIC %python
# MAGIC %timeit pagecountsEnAllDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Cache the DataFrame.  An action will materialize the cache (store it on the Spark cluster).

# COMMAND ----------

# MAGIC %python
# MAGIC (pagecountsEnAllDF
# MAGIC   .cache()         # Mark the DataFrame as cached
# MAGIC   .count()         # Materialize the cache
# MAGIC ) 

# COMMAND ----------

# MAGIC %md
# MAGIC Perform the same operation on the cached DataFrame.

# COMMAND ----------

# MAGIC %python
# MAGIC %timeit pagecountsEnAllDF.count()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC What was the change?  Now unpersist the DataFrame.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Running these commands multiple times can result in different performance.  This is due to a difference between Databricks Runtime and open source Spark.  In brief, Databricks dynamically caches remote data for you in order to alleviate IO bottlenecks.  <a href="https://databricks.com/blog/2018/01/09/databricks-cache-boosts-apache-spark-performance.html" target="_blank">See the Databricks blog for more details</a>

# COMMAND ----------

# MAGIC %python
# MAGIC pagecountsEnAllDF.unpersist()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Cluster Configuration
# MAGIC 
# MAGIC Choosing the optimal cluster for a given workload depends on a variety of factors.  Some general rules of thumb include:<br><br>
# MAGIC 
# MAGIC * **Fewer, large instances** are better than more, smaller instances since it reduces network shuffle
# MAGIC * With jobs that have varying data size, **autoscale the cluster** to elastically vary the size of the cluster
# MAGIC * Price sensitive solutions can use **spot pricing resources** at first, falling back to on demand resources when spot prices are unavailable
# MAGIC * Run a job with a small cluster to get an idea of the number of tasks, then choose a cluster whose **number of cores is a multiple of those tasks**
# MAGIC * Production jobs should take place on **isolated, new clusters**
# MAGIC * **Colocate** the cluster in the same region and availability zone as your data
# MAGIC 
# MAGIC Available resources are generally compute, memory, or storage optimized.  Normally start with compute-optimized clusters and fall back to the others based on demands of the workload (e.g. storage optimization for analytics or memory optimized for iterative algorithms such as machine learning).
# MAGIC 
# MAGIC <img alt="Best Practice" title="Best Practice" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-blue-ribbon.svg"/> When running a job, examine the CPU and network traffic using the Spark UI to confirm the demands of your job and adjust the cluster configuration accordingly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 1: Comparing Compression Algorithms
# MAGIC 
# MAGIC Compare algorithms on compression ratio and time to compress for different file types and number of partitions.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Create and Cache a DataFrame for Analysis
# MAGIC 
# MAGIC Create a DataFrame `pagecountsEnAllDF`, cache it, and perform a count to realize the cache.

# COMMAND ----------

pagecountsEnAllDF = spark.read.parquet("/mnt/training/wikipedia/pagecounts/staging_parquet_en_only_clean/")
pagecountsEnAllDF.cache()
pagecountsEnAllDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Examine the Functions for Comparing Writes and Reads
# MAGIC 
# MAGIC Examine the following functions, defined for you, for comparing write and read times by file type, number of partitions, and compression type.

# COMMAND ----------

from time import time

def write_read_time(df, file_type, partitions=1, compression=None, outputPath=username + "/academy/comparisonTest"):
  '''
  Prints write time and read time for a given DataFrame with given params
  '''
  start_time = time()
  _df = df.repartition(partitions).write.mode("OVERWRITE")
  
  if compression:
    _df = _df.option("compression", compression)
  if file_type == "csv":
    _df.csv(outputPath)
  elif file_type == "parquet":
    _df.parquet(outputPath)
    
  total_time = round(time() - start_time, 1)
  print("Save time of {}s for\tfile_type: {}\tpartitions: {}\tcompression: {}".format(total_time, file_type, partitions, compression))
  
  start_time = time()
  if file_type == "csv":
    spark.read.csv(outputPath).count()
  elif file_type == "parquet":
    spark.read.parquet(outputPath).count()
    
  total_time = round(time() - start_time, 2)
  print("\tRead time of {}s".format(total_time))
  
  
def time_all(df, file_type_list=["csv", "parquet"], partitions_list=[1, 16, 32, 64], compression_list=[None, "gzip", "snappy"]):
  '''
  Wrapper function for write_read_time() to gridsearch lists of file types, partitions, and compression types
  '''
  for file_type in file_type_list:
    for partitions in partitions_list:
      for compression in compression_list:
        write_read_time(df, file_type, partitions, compression)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Examine the Output
# MAGIC 
# MAGIC Apply `time_all()` to `pagecountsEnAllDF` and examine the results.  Why do you see these changes across different file types, partition numbers, and compression algorithms?

# COMMAND ----------

time_all(pagecountsEnAllDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review
# MAGIC **Question:** Why does compression matter when storage space is so inexpensive?  
# MAGIC **Answer:** Compression matters in big data environments largely due to the IO bound nature of workloads.  Compression allows for less data transfer across the network, speeding up tasks significantly.  It can also reduce storage costs substantially as data size grows.
# MAGIC 
# MAGIC **Question:** What should or shouldn't be compressed?  
# MAGIC **Answer:** One best practice would be to compress all files using snappy, which balances compute and compression ratio trade-offs nicely.  Compression depends largely on the type of data being compressed so a text file type like CSV will likely compress significantly more than a parquet file of integers, for example, since parquet will store them as binary by default.
# MAGIC 
# MAGIC **Question:** When should I cache my data?  
# MAGIC **Answer:** Caching should take place with any iterative workload where you read data multiple times.  Reexamining when you cache your data often leads to performance improvements since poor caching can have negative downstream effects.
# MAGIC 
# MAGIC **Question:** What kind of cluster should I use?  
# MAGIC **Answer:** Choosing a cluster depends largely on workload but a good rule of thumb is to use larger and fewer compute-optimized machines at first.  You can then tune the cluster size and type depending on the workload.  Autoscaling also allows for dynamic resource allocation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC [Start the capstone project]($./07-Capstone-Project ).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Topics & Resources
# MAGIC 
# MAGIC **Q:** Where can I get more information on autoscaling clusters?  
# MAGIC **A:** Check out the Databricks blob post <a href="https://databricks.com/blog/2018/05/02/introducing-databricks-optimized-auto-scaling.html" target="_blank">Introducing Databricks Optimized Autoscaling</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2019 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>