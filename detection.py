import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from collections import Counter
from PIL import Image
import cv2
import os

# Determines whether the fruit is Fresh or Rotten
def determineFreshOrRotten(img, fruit):

    # Image Transform
    image = Image.open(img).convert('RGB')
    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
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
        state_dict_loaded = torch.load('./fr_apple_model_weights.pth')
    elif (fruit == "banana"):
        state_dict_loaded = torch.load('./fr_banana_model_weights.pth')
    else:
        state_dict_loaded = torch.load('./fr_orange_model_weights.pth')

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

def detect_objects():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    model.eval()

    ig = Image.open(path)
    transform = T.ToTensor()
    img = transform(ig)
    with torch.no_grad():
        pred = model([img])
    bboxes, labels , scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
    print(scores)
    print(labels)
    num = torch.argwhere(scores > 0.7).shape[0]

    coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
    "frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
    "baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
    "plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
    "banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
    "pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
    "mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
    "laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
    "oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
    "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]

    chosen_classes = ["apple", "orange", "banana"]

    numImgs = 0
    cnt = Counter()

    igg = cv2.imread(path)
    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
        class_name = coco_names[labels.numpy()[i]-1]
        print(class_name)
        if (class_name in chosen_classes):

            cropped_img = igg[y1:y2, x1:x2]
            numImgs = numImgs + 1
            image_path = path_folder + 'Image_' + str(numImgs) + '.jpeg'
            cv2.imwrite(image_path, cropped_img)

            finalClassification = determineFreshOrRotten(image_path, class_name)

            print(finalClassification)

            cnt[finalClassification] += 1
            # igg = cv2.rectangle(igg, (x1, y1) , (x2, y2), (0, 255, 0), 1)
            # igg = cv2.putText(igg, class_name, (x1 , y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # cv2.imshow('' , igg)
    # cv2.waitKey(0)
    for i in range(numImgs):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        image_path = '\\fridge_images\\Image_' + str(i + 1) + '.jpeg'
        os.remove(dir_path + image_path)
    return cnt

device = 'cpu'
path = './fridge_images/original.jpeg'
path_folder = './fridge_images/'
# cnt = detect_objects()
# print(cnt)
