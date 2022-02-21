import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt




class ImageDataset:

    ''' Please provide the following structure of image dataset directory:
        image_dataset:
                - class 1:
                    - image 1;
                    - image 2;
                      ....
                - class 2
                - class 3
                  .....
        Note: "image_dataset", "class 1", "class 2", "class 3", "image 1", "image 3"
              are not compulsory names and utilized for example.'''

    def __init__(self, image_dataset_path):
        self.image_dataset_path = image_dataset_path
        self.image_dataset_content_info = {}
        self.number_of_classes = 0
        self.number_of_images = 0
        self._get_content_info()


    def _get_content_info(self):

        '''Get the structure info of given image dataset.'''

        assert os.path.isdir(self.image_dataset_path) == True, "Given directory path is not valid!"
        image_images_classes = os.listdir(self.image_dataset_path)
        assert len(image_images_classes) > 0 , "No data is found in given directory!"    
        self.number_of_classes = len(image_images_classes)
        self.number_of_images = 0
        print("Analyzing dataset's content... ")
        for image_images_class in tqdm(image_images_classes):
            image_images = os.listdir(os.path.join(self.image_dataset_path, image_images_class))
            assert len(image_images) > 0, f"Class with zero images is found, class: {image_images_class}"
            self.number_of_images += len(image_images)
            self.image_dataset_content_info[image_images_class] = {"images_number": len(image_images), "images": image_images}
    

    def mean_images_per_class(self):

        '''Calculate mean amount of images per class in the given dataset.'''

        return int(self.number_of_images / self.number_of_classes)
    

    def max_images_per_class(self):

        ''' Search for maximum number of images per class.
            Return value: maximum number of images per class.'''

        max_value = 0
        for content_info_item in self.image_dataset_content_info.items():
            max_value = max(content_info_item[1]["images_number"], max_value)
        return max_value
    

    def min_images_per_class(self):

        ''' Search for minimum number of images per class.
            Return value: minimum number of images per class.'''

        min_value = self.number_of_images
        for content_info_item in self.image_dataset_content_info.items():
            min_value = min(content_info_item[1]["images_number"], min_value)
        return min_value
    

    def get_classes_with_n_images(self, n=1):

        ''' Search for classes with number of images equal to n.

            Parameters:
                n: int
                    Number of images. By default it equals to 1.

            Return value: list of classes with n images.'''

        assert type(n) == int and n > 0, "Value of n must be positive integer!"
        classes = []
        for content_info_item in self.image_dataset_content_info.items():
            if content_info_item[1]["images_number"] == n:
                classes.append(content_info_item[0])
        return classes

    
    def proportion_of_classes_with_n_images(self, images_number, plot_pie_chart=False):

        ''' Analyzing the proportion of classes with given number of images with respect to all classes.
            
            Parameters:
                images_number: list or int
                    Number(s) of images, that classes should have during proportion analysis.
                plot_pie_chart: bool
                    A flag to decide whether or not to show the proportions on pie chart.'''            

        if type(images_number) is not list: images_number = [images_number] 
        assert sum([(i>0 and type(i) == int) for i in images_number]) == len(images_number), "Invalid images_number, please provide only positive integers!"
        sizes = np.zeros(len(images_number)).astype(int)
        for content_info_item in self.image_dataset_content_info.items():
            if content_info_item[1]["images_number"] in images_number:
                sizes[images_number.index(content_info_item[1]["images_number"])] += 1
        sizes = np.append(sizes, self.number_of_classes - np.sum(sizes))       
        
        for i, size in enumerate(sizes[:-1]):
            print(f"Number of classes with only {images_number[i]} images : {size}")
        print(f"Remaining number of classes : {sizes[-1]}")

        if plot_pie_chart:        
            labels = []
            for image_number in images_number:
                labels.append(f"classes with {image_number} imgs")
            labels.append("Other classes")
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            fig1.subplots_adjust(0.3,0,1,1)
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                    startangle=90)                 

            ax1.axis('equal')
            
            plt.show()


    def _write_datalist_to_file(self, file_path, datalist):

        ''' Writing data into file.

            Parameters:
                file_path: str
                    Path of file to write data.
                datalist: list
                    List of items to write into file.'''

        assert os.path.isfile(file_path) == True, f"{file_path=} is not a file!"
        with open(file_path, "w") as datalist_file:
            for datalist_item in datalist:
                datalist_file.write(datalist_item + "\n")
            
    
    def create_train_test_list(self, n=3, train_list_file_path=None, test_list_file_path=None,
                               full_list_file_path=None, plot_relation=False):
    
        ''' Create train and test lists for identification task.
            Dataset split is done as following:
                for every class:
                    if number of images >= every_n parameter
                        then select first int(number_of_images/n) items for test, others for train
                    else skip the class.
            
            Parameters:
                n: int
                    This variale is used to define the number of test images in every class by 
                    following expression: int(number_of_images/n). By default the value is equal to 3.
                train_list_file_path: str
                    Path to file for recording train items of image dataset.
                test_list_file_path: str
                    Path to file for recording test items of image dataset.
                full_list_file_path: str
                    Path to file for recording full (i.e. train and test) items of image dataset.
                plot_relation: bool
                    A flag to decide whether or not to show the proportion of
                    train and test dataset items on pie chart.
            
            Return value: a tuple of lists of training, test, full items of image dataset.'''

        assert type(n) == int and n > 0, "Invalid every_n parameter is given!"
        train_list = []
        test_list = []
        full_list = []
        class_counter = 0
        for content_info_item in self.image_dataset_content_info.items():
            if content_info_item[1]["images_number"] >= n:
                test_images_number = int(content_info_item[1]["images_number"] / n)
                for i, image in enumerate(content_info_item[1]["images"]):
                    list_item = content_info_item[0] + "/" + image + " " + str(class_counter)
                    if i < test_images_number:
                        test_list.append(list_item)
                    else:
                        train_list.append(list_item)
                    full_list.append(list_item)
                class_counter += 1            

        if train_list_file_path is not None:
            self._write_datalist_to_file(train_list_file_path, train_list)
                    
        if test_list_file_path is not None:
            self._write_datalist_to_file(test_list_file_path, test_list)
            
        if full_list_file_path is not None:
            self._write_datalist_to_file(full_list_file_path, full_list)
        
        if plot_relation:        
            print(f"Number of classes: {class_counter}")
            print(f"Number of train images: {len(train_list)}")
            print(f"Number of test images: {len(test_list)}")
            print(f"Total number of images: {len(full_list)}")
            
            labels = ["Train", "Test"]
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            fig1.subplots_adjust(0.3,0,1,1)
            sizes = [len(train_list), len(test_list)]
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                    startangle=90,)
            ax1.axis('equal')
            plt.show()
            
        return (train_list, test_list, full_list)
    
    
    def get_class_info(self, name):

        ''' Get class info with given id.
            
            Parameter: 
                name: str
                    Name of class.
            
            Return value: information about class containing images_number, images and etc.'''

        assert type(name) == str, "Invalid name value is given!" 
        assert name in self.image_dataset_content_info.keys(), "Class with given name doesn't exist!"
        return self.image_dataset_content_info[name]


    def analyze(self, images_number=[1, 2, 3]):

        '''Function analyzes image dataset for :
                - number of images and classes;
                - mean number of images per class;
                - minimum number of images per class;
                - maximum number of images per class;
                - proportion of classes with given number(s) of images;
                
            Parameters:
                images_number: list or int
                    Number(s) of images, that classes should have during proportion analysis.'''
                
        print(f"Number of images in image dataset: {self.number_of_images}")
        print(f"Number of classes in image dataset: {self.number_of_classes}")
        print(f"Mean number of images per class: {self.mean_images_per_class()}")
        print(f"Minimum number of images per class: {self.min_images_per_class()}")
        print(f"Maximum number of images per class: {self.max_images_per_class()}")
        self.proportion_of_classes_with_n_images(images_number, plot_pie_chart = True)   
    
    