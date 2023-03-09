import math
import cv2
import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from os import listdir
from collections import Counter
from sklearn.cluster import DBSCAN
from torchvision.io import read_image
from PIL import Image

# Determines the fruit type of the given image, from Apple, Banana, Orange
def determineFruitType(img):
    
    # Image Transform
    image = Image.open(img).convert('RGB')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)

    # Model
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights).to(device)

    # Modify fc layer
    model.fc = nn.Sequential(nn.Linear(512, 3)).to(device)

    # Load model weights
    state_dict_loaded = torch.load(root + './fruit_type_model_weights.pth')
    model.load_state_dict(state_dict_loaded)
    model.eval()

    # Evaluate
    with torch.no_grad():
        output = model(image)

    # Get the predicted class
    prediction = torch.argmax(output, dim = 1)
    fruit_type = fruit_types[prediction]
    return fruit_type

# Determines whether the fruit is Fresh or Rotten
def determineFreshOrRotten(img, fruit):

    # Image Transform
    image = Image.open(img).convert('RGB')
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    image = transform(image)
    image = image.unsqueeze(0)

    # Model
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        # nn.Sigmoid(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        # nn.Sigmoid(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        # nn.Sigmoid(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        # nn.Sigmoid(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(16384, 4096),
        nn.ReLU(inplace=True),
        # nn.Sigmoid(),
        nn.Dropout(p=0.5),

        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        # nn.Sigmoid(),
        nn.Dropout(p=0.5),

        nn.Linear(4096, 1)
    )
    model.to(device)

    # Load model weights
    if (fruit == "apple"):
        state_dict_loaded = torch.load(root + './fr_apple_model_weights.pth')
    elif (fruit == "banana"):
        state_dict_loaded = torch.load(root + './fr_banana_model_weights.pth')
    else:
        state_dict_loaded = torch.load(root + './fr_orange_model_weights.pth')

    model.load_state_dict(state_dict_loaded)
    model.eval()

    # Evaluate
    with torch.no_grad():
        output = model(image)

    # Get the predicted class
    outputProb = output.detach().numpy()[0]
    
    if (outputProb > 0):
        fr = "fresh"
    else:
        fr = "rotten"

    return fr + " " + fruit

# Creates bounding boxes on the image and runs the ML models on each box
# Returns a count of each category within the image
def makeBoundingBoxes():

    path = root + img_folder

    # Load the image and convert it to grayscale
    img = cv2.imread(path + img_name)
    imgDup = cv2.imread(path + img_name)
    dimentions = img.shape
    imgXDim = dimentions[0]
    imgYDim = dimentions[1]
    imgDist = math.sqrt(math.pow(imgXDim, 2) + math.pow(imgYDim, 2))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blurring to reduce noise
    # blurred = cv2.GaussianBlur(gray, (13, 13), 0)     # for clearer images
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 50, 100)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the bounding boxes for each contour
    boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # Cluster the bounding boxes using DBSCAN
    X = np.array(boxes)
    clustering = DBSCAN(eps = 10, min_samples = 1).fit(X)

    numImgs = 0
    cnt = Counter()

    # Draw a single bounding box around each group of objects
    for i in np.unique(clustering.labels_):
        if i == -1:
            # Skip noise points (points not assigned to a cluster)
            continue
        group_boxes = X[clustering.labels_ == i]

        x_min, y_min, x_max, y_max = np.min(group_boxes[:, 0]), np.min(group_boxes[:, 1]), np.max(group_boxes[:, 0] + group_boxes[:, 2]), np.max(group_boxes[:, 1] + group_boxes[:, 3])
        
        xDim = x_max - x_min
        yDim = y_max - y_min
        dist = math.sqrt(math.pow(xDim, 2) + math.pow(yDim, 2))

        if (xDim < imgXDim * 0.2 and yDim < imgYDim * 0.2):
            continue

        if (xDim < imgXDim * 0.05 or yDim < imgYDim * 0.05):
            continue
        
        cropped_img = imgDup[y_min:y_max, x_min:x_max]

        # Remove the box if it contains over 50% white pixels
        white_pixels = cv2.inRange(cropped_img, (200, 200, 200), (255, 255, 255))
        num_white_pixels = cv2.countNonZero(white_pixels)
        num_total_pixels = cropped_img.shape[0] * cropped_img.shape[1]
        white_pixel_percentage = num_white_pixels / num_total_pixels
        
        if white_pixel_percentage > 0.25:
            continue

        numImgs = numImgs + 1

        image_path = path + 'Image_' + str(numImgs) + '.jpeg'

        cv2.imwrite(image_path, cropped_img)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        fruitType = determineFruitType(image_path)
        finalClassification = determineFreshOrRotten(image_path, fruitType)

        print(finalClassification)

        cnt[finalClassification] += 1

    # Display the image with bounding boxes
    cv2.imwrite(path + 'BoxedImage.jpeg', img)
    # cv2.imshow("Fruits with Bounding Boxes", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Delete all images
    # for images in os.listdir(path):
    #     os.remove(os.path.join(path, images))

    return cnt

# Global Variables
# root = "C:\\Users\\kunal\\OneDrive\\Desktop\\School\\FYDP\\Code\\"
root = 'C:\\Users\\Adrian\\Desktop\\laptop_server\\'
img_folder = "fridge_images\\"
img_name = "original.jpeg"
device = 'cpu'

fruit_types = ["apple", "banana", "orange"]

# Run
# fruitCounts = makeBoundingBoxes()

# print(fruitCounts)
