# H5DAT
HDF5 Data Access Toolkit for Python 

The overall design of the processing uses batches in combination with the multi-processing(mp) 
Python module to manage processes and resources. Each content "classification" is contained within 
it's own class, e.g. **ImageProcessor, VideoProcessor, TextFileProcessor**, etc. 

Content types are mapped to each processor, such as JSON/CSV to the TextFileProcessor, image formats
to the ImageProcessor, and so on. The **DataProcessor** class within the dataWrangler modules is the I/O
handler and processing control between different content types. It also handles the multi-processing functions.

The simple overview process is below

    input file --> read--> convert --> publish tuples to mp.queue --> deque --> write to HDF5

The initialization of the classes does require parameters, which is not ideal. It was done this way due
to file locking issues when reading/writing to the same HDF5 file with functions only. I would like to
improve this in the future. The provided samples are a good reference for how to use the classes.

The contained code is an attempt at creating some processing structure when dealing with HDF5 files.
In simple terms its an I/O toolkit that reads from different file formats(zip, gzip, tar/tar.gz)
without completely decompressing the file archive and converts specified file types(images, video,
json, csv) to datasets and stores them in a HDF5 file destination.

The principle users would be ML Engineers, Data Engineers, and AI scientists that need to store
and classify various inputs. JSON and CSV files are converted to key/value pair strings and stored
as datasets. Various image and video files are converted to Numpy arrays(ndarray) and datasets are
created.

The configuration for overall use and optional schema definitions are managed via json files. 
The "samples" folder contains examples for each and scripts on sample usage.

Docs with more detail on classes and functions can be found:

[dataWrangler](hdfa/src/hdfa/processors/htmldoc/dataWrangler.html)

[dataRetriever](hdfa/src/hdfa/processors/htmldoc/dataRetriever.html)

[imageProcessor](hdfa/src/hdfa/processors/htmldoc/imageProcessor.html)

[videoProcessor](hdfa/src/hdfa/processors/htmldoc/videoProcessor.html)

[schemaProcessor](hdfa/src/hdfa/processors/htmldoc/schemaProcessor.html)

[parsingProcessor](hdfa/src/hdfa/processors/htmldoc/parsingProcessor.html)

[textFileProcessor](hdfa/src/hdfa/processors/htmldoc/textFileProcessor.html)

[mpLocal](hdfa/src/hdfa/processors/htmldoc/mpLocal.html)

***Testing***

Functional testing has been done with the content types that are supported. In addition
there was some throughput testing with the [Fashion-Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
sourced from Kaggle. That particular dataset was downloaded, and archives created in both zip and tarfile gzip (tar.gz) formats. The 
multi-processing,buffering, and batching proved effective as the 177k+ files(jpeg, csv, json) were converted(numpy, strings) to datasets(batches of 5k). 
It was a single execution on a MacBook Pro(2024): 14-inch M4 Pro(12 cores), 24GB RAM, 500GB internal + 2TB external storage,
with a runtime of 34 hours 21 minutes. The execution consumed one cpu core the entire time, using approx. 1.6-2G of RAM, but
up to 80GB of VM. The resulting file size was approx. 174GB.

A custom schema input was also used(samples/file_schema.json), in order to classify the files by file type. It provides a basic
mapping mechanism for taking an input parameter and mapping it to a group within the destination HDF5. 

