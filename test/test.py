import unittest
from image_dataset_analysis import ImageDataset



class TestImageDatasetAnalyzer(unittest.TestCase):
    image_dataset = ImageDataset("C:\\Users\\Samandar\\Desktop\\lfw_aligned_128\\lfw_aligned_128")
    
    def test_get_content_info(self):
        self.assertEqual(self.image_dataset.number_of_classes, 5734)
    
    def test_mean_value(self):
        self.assertEqual(self.image_dataset.mean_images_per_class(), 2)

    def test_min_images_per_class(self):
        self.assertEqual(self.image_dataset.min_images_per_class(), 1)

    def test_max_images_per_class(self):
        self.assertEqual(self.image_dataset.max_images_per_class(), 530)

    def test_get_classes_with_n_images(self):
        self.assertEqual(self.image_dataset.get_classes_with_n_images(n=530)[0],
                         'George_W_Bush')
    
    def test_create_train_test_list(self):
        train, test, full = self.image_dataset.create_train_test_list(n=3)
        self.assertEqual(len(train), 5297)
        self.assertEqual(len(test), 2306)
        self.assertEqual(len(full), 7603)
    
    def test_get_class_info(self):
        class_info = self.image_dataset.get_class_info('George_W_Bush')
        self.assertEqual(class_info["images_number"], 530)
        self.assertEqual(class_info["images"][0], 'George_W_Bush_0001.jpg')

if __name__ == '__main__':
    unittest.main()



