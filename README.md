# Image dataset analysis

In the field of Computer Vision, image datasets play one of the important roles. Everyone, who begin his/her research in the aforementioned sphere, initially, need to choose model and set of data, appropriate for the given CV task. Usually, it is useful to make analysis (ex. analyze proportion of classes with given number of images or get classes with n images) in order to take a closer look at data. In this repository, you might find useful tools to perform image dataset analysis.


## Let's analyze the image dataset

>Let's choose the [LFW (Labelled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/) face database as an example dataset.


1) First of all, it is necessary to create an instance of ImageDataset class and pass the path of dataset's directory as an argument.

   >Following structure of dataset directory is required to analyze it appropriately:
    - image_dataset:
        - class 1:
            - image 1;
            - image 2;
            - ....
        - class 2
        - class 3
        - .....
   
   
   ```   
   from image_dataset_analysis import ImageDataset
   lfw = ImageDataset("/LFW")   
   ```
2) Now, we can get some useful information about dataset by calling the following method:
   
   ```
   lfw.analyze()
   ```
   
   Output:
   
   ```
      sNumber of images in image dataset: 13214
      Number of classes in image dataset: 5734
      Mean number of images per class: 2
      Minimum number of images per class: 1
      Maximum number of images per class: 530
      Formats of images in dataset:
         jpg: 13213 (100.0%)
      Number of classes with only 1 images : 4057
      Number of classes with only 2 images : 777
      Number of classes with only 3 images : 290
      Remaining number of classes : 610s
   ```
   ![proportion of images](/example/images/analyze_output.png)
   
## Do you want to know more?
You might get useful information about other valuable tools along with their implementations in the [example.pdf](https://github.com/SamandarYokubov/image_dataset_analysis/blob/main/example/example.pdf) file.
In addition, you may use [example.ipynb](https://github.com/SamandarYokubov/image_dataset_analysis/blob/main/example.ipynb) to make analysis of your image datasets. By the way, don't forget to change the path of dataset's directory to your own one!

##  Any questions :question:
Feel free to ask or express your ideas in [issues](https://github.com/SamandarYokubov/image_dataset_analysis/issues) section.

