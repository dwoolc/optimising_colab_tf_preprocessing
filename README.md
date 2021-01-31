# optimising_colab_tf_preprocessing
A short git outlining optimsing data load and pre-process using Colab &amp; Tensorflow.

Full write-up available <a href="#">here</a>.

Core concepts are to:
- load zipped data to the colab instance for each run to reduce access times
- unzip the data
- use efficient preprocessing such as prefetch, mapping and caching in combination to provide multiple orders of magnitude performance improvements.

Main ideas taken from the official Tensorflow Guide <a href="https://www.tensorflow.org/guide/data_performance">here</a>.
