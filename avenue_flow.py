import glob
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torchvision.transforms as T
#import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
from torchvision.io import write_jpeg
from tqdm import tqdm
import cv2

# If you can, run this example on a GPU, it will be a lot faster.
device = "cpu"

data_path = '/local/scratch/irr_prediction/datasets/avenue'
train_path = data_path + '/training_videos'
test_path = data_path + '/testing_videos'

def read_tif_image(image_path):
    '''
    read a tiff image using pillow and convert it to a tensor
    '''
    image = Image.open(image_path)
    tensor = ToTensor()(image)
    return tensor

def read_avi(path):
    '''
    read an avi video using opencv and convert it to a tensor
    '''
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        # ret is boolean indicating if the frame was read correctly
        # frame has shape (height, width, channels) 
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    frames_tensor = torch.stack([torch.tensor(frame) for frame in frames]).permute(0, 3, 1, 2)
    return frames_tensor

def preprocess(batch):
    '''
    Preprocess: convert the image to float32, normalize it to [-1, 1], resize it to (240, 360),
    and repeat the grayscale image 3 times to make it compatible with the RGB model
    '''
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5], std=[0.5]),  # map [0, 1] into [-1, 1]
            T.Resize(size=(240, 360)),
        ]
    )
    batch = transforms(batch)
    # repeat the grayscale image 3 times to make it compatible with the RGB model
    #batch = batch.repeat(1, 3, 1, 1)
    return batch

def compute_flows(img1_batch, img2_batch):
    print(f"Computing flows for {img1_batch.shape[0]} images")
    img1_batch = preprocess(img1_batch).to(device)
    img2_batch = preprocess(img2_batch).to(device)

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()
    flows = []
    for i in range(len(img1_batch)):
        flow = model(img1_batch[i:i+1], img2_batch[i:i+1])
        predicted_flows = flow[-1]
        predicted_flows = predicted_flows.detach().cpu()
        flows.append(predicted_flows)

    flows = torch.stack(flows).squeeze()
    return flows

def construct_batches(frames):
    '''
    construct to batches of images from the frames

    frames: tensor of shape (N, C, H, W)
    returns: two batches of images of shape (N-1, C, H, W)  
    '''
    img1_batch = frames[:-1]
    img2_batch = frames[1:]
    return img1_batch, img2_batch

def save_flows(path, predicted_flows):
    '''
    Save the flow images as NPZ
    '''
    for i, flow_img in tqdm(enumerate(predicted_flows), total=len(predicted_flows)):
        flow_img_np = flow_img.detach().cpu().numpy()
        n = str(i+1).zfill(3)
        np.savez(os.path.join(path, f"{n}.npz"), flow_img_np)
    print(f"Saved to {path}/***.npz")

def save_flows_combined(path, train_case,  predicted_flows):
    '''
    Save the flow images as NPZ
    '''
    case_nr = os.path.basename(os.path.normpath(train_case))
    flow_img_np = np.stack([flow_img.detach().cpu().numpy() for flow_img in predicted_flows])
    np.savez(os.path.join(path, f"{case_nr}.npz"), flow_img_np)
    print(f"Saved to {path}")

def save_flow_imgs(path, predicted_flows):
    # Save the flow images as JPEG
    for i, predicted_flow in enumerate(predicted_flows):
        flow_img = flow_to_image(predicted_flow).to("cpu")
        n = str(i+1).zfill(3)
        write_jpeg(flow_img, os.path.join(path, f"predicted_flow_{n}.jpg"))


#train_cases = sorted([os.path.join(train_path, f) for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))])
#test_cases = sorted([os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, f))])
test_cases = []

train_cases = glob.glob(train_path + '/*.avi')

# TODO: only one train case for testing
train_cases = train_cases[:1]

for train_case in tqdm(train_cases, total=len(train_cases)):
    print(f"Train case: {train_case}")
    #train_case_files = sorted([os.path.join(train_case, f) for f in os.listdir(train_case) if f.endswith('.tif')])
    frames = read_avi(train_case)
    img1_batch, img2_batch = construct_batches(frames)

    # TODO: only use the first two frames for testing
    img1_batch = img1_batch[:2]
    img2_batch = img2_batch[:2]

    predicted_flows = compute_flows(img1_batch, img2_batch)
    save_flows_combined(train_path, train_case, predicted_flows)
    save_flow_imgs(train_case, predicted_flows)

for test_case in tqdm(test_cases, total=len(test_cases)):
    print(f"Test case: {test_case}")
    test_case_files = sorted([os.path.join(test_case, f) for f in os.listdir(test_case) if f.endswith('.tif')])
    if len(test_case_files) < 2: continue
    img1_batch, img2_batch = construct_batches(test_case_files)
    predicted_flows = compute_flows(img1_batch, img2_batch)
    save_flows(test_case, predicted_flows)


### Stacking the flows for each train case
# pytorch outputs the flows in the shape (2,H,W)
# the projects code expects the flows in the shape (N_samples, H, W, 2)
# for train_case in tqdm(train_cases, desc="Processing train cases"):
#         case_name = os.path.basename(train_case)
#         npz_files = sorted([os.path.join(train_case, f) for f in os.listdir(train_case) if f.endswith('.npz')])

#         stacked_flows = []
#         for npz_file in npz_files:
#             data = np.load(npz_file)
#             flow_img = data['arr_0']
#             flow_img = np.transpose(flow_img, (1, 2, 0))
#             stacked_flows.append(flow_img)

#         stacked_flows_np = np.stack(stacked_flows)
#         output_file = os.path.join(train_path, f"{case_name}.npz")
#         np.savez(output_file, data=stacked_flows_np)
#         print(f"Saved stacked flows to {output_file}")

## transposing the flows and saving as npy instead of npz for MULDE
for test_case in tqdm(test_cases, desc="Processing train cases"):
        case_name = os.path.basename(test_case)
        npz_files = sorted([os.path.join(test_case, f) for f in os.listdir(test_case) if f.endswith('.npz')])

        for npz_file in npz_files:
            data = np.load(npz_file)
            flow_img = data['arr_0']
            flow_img = np.transpose(flow_img, (1, 2, 0))

            output_file = npz_file.replace('.npz', '.npy')
            np.save(output_file, flow_img)
            print(f"Saved flow to {output_file}")
