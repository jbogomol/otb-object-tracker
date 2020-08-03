# train.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/28/2020
#
# Trains a convolutional neural network to guess frame-to-frame object motion
# in x and y direction.
# Formulated as a classification problem with a cross-entropy loss function
#
# Before running
# Run create_images.py to create necessary data and csv files containing
# ground truths.


""" IMPORTS """
import networks
import datasets
import func_file
import func_tensor
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import os


""" GLOBALS """
# on server or local computer
on_server = torch.cuda.is_available()

# using gpu or not?
use_gpu = True and on_server

# which gpu
if on_server:
    torch.cuda.set_device(0)

# directory with images and results csv files
if on_server:
    datadir = "/home/datasets/data_jbogomol/OTB_data/results/"
else:
    datadir = "../OTB_data/results/"

# directory to store saves
reportdir = "./report/"

# path to network to save
netpath = os.path.join(reportdir, "otb_net.pth")


""" SETUP """
# keep same random seed for replicable results
random_seed = 1
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# paths to csv files
csv_path_list = func_file.get_files_sorted(directory=datadir, extension=".csv")
np.random.shuffle(csv_path_list)
 
# create pytorch datasets for train, validation, test. split by video
n_videos = len(csv_path_list)
n_videos_test = n_videos // 3
n_videos_validation = (n_videos - n_videos_test) // 6
n_videos_train = n_videos - n_videos_test - n_videos_validation

# initialize empty sets
train_set = torchvision.datasets.FakeData(size=0)
validation_set = torchvision.datasets.FakeData(size=0)
test_set = torchvision.datasets.FakeData(size=0)

# fill datasets video by video, looping through all csv paths
for i, csv_path in enumerate(csv_path_list):
    # get set for current csv path
    set_i = datasets.TrackingDataset(csv_path=csv_path)

    # determine where to put set
    if i < n_videos_train:
        train_set = torch.utils.data.dataset.ConcatDataset(
                        [train_set, set_i])
    elif i < n_videos_train + n_videos_validation:
        validation_set = torch.utils.data.dataset.ConcatDataset(
                        [validation_set, set_i])
    else:
        test_set = torch.utils.data.dataset.ConcatDataset(
                        [test_set, set_i])

# print dataset info
print("DATASET INFO")
print("total videos: ", n_videos)
print("# training videos:   ", n_videos_train)
print("# validation videos: ", n_videos_validation)
print("# testing videos:    ", n_videos_test)
n_train = len(train_set)
n_validation = len(validation_set)
n_test = len(test_set)
print("image pairs in train set:      ", n_train)
print("image pairs in validation set: ", n_validation)
print("image pairs in test set:       ", n_test, "\n")

# network hyperparameters
n_epochs = 50
learning_rate = 0.001
momentum = 0.9
batch_size_train = 64
batch_size_validation = 1
batch_size_test = 1
log_interval = 10 # print loss every log_interval mini batches

# dataloaders
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size_train, shuffle=True)
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=batch_size_validation, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size_test, shuffle=True)


# check shapes
imgs, v = iter(train_loader).next()
print("INPUT SHAPES")
print("imgs.shape:", imgs.shape)
print("v.shape:", v.shape, "\n")


# init network and optimizer
network = networks.NetworkClassifier()
if use_gpu:
    network = network.cuda()
optimizer = optim.SGD(
    network.parameters(),
    lr=learning_rate,
    momentum=momentum)
print("NETWORK INFO")
print(network, "\n")


""" TRAINING LOOP """
print("BEGIN TRAINING LOOP")
for epoch in range(n_epochs):
    running_loss = 0.0
    for batch_index, batch in enumerate(train_loader, 0):
        # get inputs and labels
        inputs, labels = batch
        
        # reformat labels
        labels = labels + 32
        labels = labels.long()
        labels_x = func_tensor.saturate_1d(
            t=labels[:,0], low=0, high=(2 * 32))
        labels_y = func_tensor.saturate_1d(
            t=labels[:,1], low=0, high=(2 * 32))
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
            labels_x = labels_x.cuda()
            labels_y = labels_y.cuda()

        # zero the gradients
        optimizer.zero_grad()

        # forward, backward, gradient descent
        outputs = network(inputs)
        outputs_x = outputs[:,0,:]
        outputs_y = outputs[:,1,:]
        loss_x = F.cross_entropy(outputs_x, labels_x)
        loss_y = F.cross_entropy(outputs_y, labels_y)
        loss = loss_x + loss_y
        loss.backward()
        optimizer.step()

        # print log
        running_loss += loss.item()
        if batch_index % log_interval == log_interval - 1:
            print("epoch %d,\tbatch %5d,\ttraining loss: %.3f" %
                  (epoch + 1, batch_index + 1, running_loss / log_interval))
            running_loss = 0.0

    # end of epoch, print log
    print("END OF EPOCH " + str(epoch + 1))

    # test on validation set
    print("Testing on validation set...")
    correct = 0
    off_by_one = 0
    total = n_validation * 2 # count x and y guesses separately
    with torch.no_grad():
        for batch in validation_loader:
            images, labels = batch
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs = network(images)
            preds = outputs.argmax(dim=2)
            labels += 32
            x_diff = preds[:,0] - labels[:,0]
            y_diff = preds[:,1] - labels[:,1]
            for i in range(batch_size_validation):
                error_x = abs(x_diff[i].item())
                error_y = abs(y_diff[i].item())

                if error_x < 1:
                    correct += 1
                elif error_x < 2:
                    off_by_one += 1

                if error_y < 1:
                    correct += 1
                elif error_y < 2:
                    off_by_one += 1
    
    print("# correct:\t" + str(correct) + "/" + str(total) + " = "
          + str(100.0 * correct / total) + "%")
    print("# off by 1:\t" + str(off_by_one) + "/" + str(total) + " = "
          + str(100.0 * off_by_one / total) + "%\n")



print("FINISHED TRAINING")
print("SAVING NETWORK")
torch.save(network.state_dict(), netpath)


















