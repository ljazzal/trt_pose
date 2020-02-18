# Human pose estimation based on Nvidia's trt_pose (https://github.com/NVIDIA-AI-IOT/trt_pose)
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch2trt
from torch2trt import TRTModule
from trt_pose import coco, models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jetcam.usb_camera import USBCamera
from jetcam.utils import bgr8_to_jpeg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import time
import signal
import sys

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
WIDTH = 224
HEIGHT = 224

global camera
global im

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def signal_handler(sig, frame):
    global camera
    print('You pressed Ctrl+C!')
    camera.unobserve_all()
    sys.exit(0)

def benchmark(frame_count=50.0):
    # Return FPS for current model
    t0 = time.time()
    torch.cuda.current_stream().synchronize()
    for i in range(frame_count):
        y = model_trt(data)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()
    return frame_count / (t1 - t0)

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def execute(change):
    image = change['new']
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    # image_w.value = bgr8_to_jpeg(image[:, ::-1, :])
    return image

def update(i):
    global im
    im.set_data(execute({'new': camera.value}))

def close(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    with open('human_pose.json', 'r') as f:
        human_pose = json.load(f)
    
    topology = coco.coco_category_to_topology(human_pose)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    # Load model
    # TODO: extend model to include additional joints
    print("Loading model")
    model = models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    model.load_state_dict(torch.load(MODEL_WEIGHTS))

    # Optimization with tensorRT
    # NOTE: optimization is device specific
    # data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
    # model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    # torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

    # Load optimized model
    print("Loading optimized model")
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))  

    # Setup camera and visuals
    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    
    camera = USBCamera(width=WIDTH, height=HEIGHT, capture_device=1)
    camera.running = True

    # Attach oberver to act on each new frame received
    # camera.observe(execute, names='value')
    
    im = plt.imshow(execute({'new': camera.value}))

    ani = FuncAnimation(plt.gcf(), update, interval=200)

    cid = plt.gcf().canvas.mpl_connect("key_press_event", close)

    plt.show()


