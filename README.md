# CaT-CAVS-Traversability-Dataset-for-Off-Road-Autonomous-Driving
In the context of autonomous driving, the existing semantic segmentation concept strongly
supports on-road driving where hard inter-class boundaries are enforced and objects can be categorized
based on their visible structures with high confidence. Due to the well-structured nature of typical onroad scenes, current road extraction processes are largely successful and most types of vehicles are able to
traverse through the area that is detected as road. However, the off-road driving domain has many additional
uncertainties such as uneven terrain structure, positive and negative obstacles, ditches, quagmires, hidden
objects, etc. making it very unstructured. Traversing through such unstructured area is constrained by a
vehicle’s type and its capability. Therefore, an alternative approach to segmentation of the off-road driving
trail is required that supports consideration of the vehicle type in a way that is not considered in state-ofthe-art on-road segmentation approaches. To overcome this limitation and facilitate the path extraction in
the off-road driving domain, we propose traversability concept and corresponding dataset which is based on
the notion that the driving trails should be finely resolved into different sub-trails and areas corresponding to
the capability of different vehicle classes in order to achieve safe traversal. Based on this, we consider three
different classes of vehicles (sedan, pickup, and off-road) and label the images corresponding to the traversing
capability of those vehicles. So the proposed dataset facilitates the segmentation of off-road driving trail
into three regions based on the nature of the driving area and vehicle capability. We call this dataset as
CaT (CAVS Traversability, where CAVS stands for Center for Advanced Vehicular Systems) dataset and is
publicly available at https://www.cavs.msstate.edu/resources/downloads/CaT/CaT.tar.gz.

# Training with CaT dataset
--Download and extract the CaT dataset from [here](https://www.cavs.msstate.edu/resources/downloads/CaT/CaT.tar.gz). Set the path before "Train" and "Test" folders inside the data pack in train.py and test.py scripts.
--Download the pretrained resnet models from [here](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) and put it in trained model/trained_models.
--Run python3 train.py while being in the main folder where the train.py file resides.
--After training is complete the final trained model is saved in logs/cat as net_latest.pth. Other intermediate models can be ignored or deleted if they are no more of use.

# Testing
--Run python3 test.py and the predicted RGB images are saved in results/cat.

# Requirements
-Ubuntu >=16.04
-Python3.6
-Pytorch >=0.4
-opencv-python, Numpy

# Acknowledgements
--The model scripts are taken from [here](https://github.com/hszhao/semseg). We are very much thankful for the authors for sharing their codes.

# Citation
If you find the dataset and the results helpful please consider citing our paper as:

@article{sharma2022cat,
  title={CaT: CAVS Traversability Dataset for Off-Road Autonomous Driving},
  author={Sharma, Suvash and Dabbiru, Lalitha and Hannis, Tyler and Mason, George and Carruth, Daniel W and Doude, Matthew and Goodin, Chris and Hudson, Christopher     and Ozier, Sam and Ball, John E and others},
  journal={IEEE Access},
  volume={10},
  pages={24759--24768},
  year={2022},
  publisher={IEEE}
}


