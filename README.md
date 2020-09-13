# otb-object-tracker
Tracks objects in the OTB visual tracking dataset.
By Jackson Bogomolny for Prof. Aswin Sankaranarayanan
Carnegie Mellon University Image Science Lab

Required:
    - numpy
    - pytorch, torchvision
    - opencv (cv2)
    - matplotlib
    - pandas
    - seaborn

How to use:
1. Create image data to train network
    Change file path globals in create_images.py to OTB data directories.
    Run:
        $ create_images.py
    Image data and csv files containing ground truths will be created.
2. Train the network
    Change file path globals in train.py to your OTB data directories.
    Ensure global variable train_on is set to True.
    Run:
        $ train.py
    Network will be trained on training data, with training loss and validation
    data scores printed to the command line. After training, the network will
    be tested on testing data, which will save error images, a heat map, and a
    map of error(x) and error(y) created in a specified error directory.
    After training once, the network can be tested again without re-training by
    setting the global variable train_on to False.

Network object printout from pytorch:
NetworkClassifier(
  (conv1): Conv2d(6, 16, kernel_size=(3, 3), stride=(2, 2))
  (conv1_bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
  (conv2_bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
  (conv3_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=61504, out_features=240, bias=True)
  (fc2): Linear(in_features=240, out_features=150, bias=True)
  (out): Linear(in_features=150, out_features=130, bias=True)
)







