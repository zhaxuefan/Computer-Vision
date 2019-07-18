# Computer-Vision
Bag of words:
Building an end to end system that will determine which type of scene it is.
![image](https://github.com/zhaxuefan/image/blob/master/718.png)
Bag of Words represents a document as a vector or histogram of counts for each word that occurs in the document, as shown in Figure 2. The hope is that different documents in the same class will have a similar collection and distribution of words, and that when we see a new document, we can find out which class it belongs to by comparing it to the histograms already in that class.
![image](https://github.com/zhaxuefan/image/blob/master/7181.png)
There is 3 major part: 
Part 1: build a dictionary of visual words from training data.
Part 2: build the recognition system using visual word dictionary and training images.
Part 3: evaluate the recognition system using test images.
In Part 1, use the provided filter bank to convert each pixel of each image into a high dimensional representation that will hopefully capture meaningful information, such as corners, edges etc. This will take each pixel from being a 3D vector of color values, to
an nD vector of filter responses. Then take these nD pixels from all of the training images and and run K-means clustering to find groups of pixels. Each resulting cluster center will become a visual word, and the whole set of cluster centers becomes our dictionary of
visual words.
![image](https://github.com/zhaxuefan/image/blob/master/7182.png)
In Part 2, the dictionary of visual word you produced will be applied to each of the training images to convert them into a wordmap. This will take each of the nD pixels in all of the filtered training images and assign each one a single integer label, corresponding to the
closest cluster center in the visual words dictionary. Then each image will be converted to a “bag of words”; a histogram of the visual words counts. Use these to build the classifier.
![image](https://github.com/zhaxuefan/image/blob/master/7183.png)
In Part 3 will evaluate the recognition system built. This will involve taking the test images and converting them to image histograms using the visual words dictionary and the function you wrote in Part 2. Next, for nearest neighbor classification,  use a
histogram distance function to compare the new test image histogram to the training image histograms in order to classify the new test image. Doing this for all the test images will give idea of how good your recognition system.

In document classification, inverse document frequency (IDF) factor is incorporated which diminishes the weight of terms that occur very frequently in the document set.Improve system by IDF.
