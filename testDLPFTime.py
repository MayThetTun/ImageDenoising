import argparse
import glob
import os
import time
import cv2
import matplotlib.pyplot
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from models import FFDNet
from utils import batch_psnr, batch_ssim, batch_fsim, batch_fsimgray, normalize, remove_dataparallel_wrapper
import lpips

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
    if dis <= r:
        return 1.0
    else:
        return 0


def mask_radial(img, r):
    bs, ch, rows, cols = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    mask = np.zeros((bs, ch, rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[:, :, i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def convertFreqImage(img):  # Convert Frequency Domain
    x = img.to('cpu').detach().numpy().copy()
    bs, c, M, N = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    # Gray (35) Color (25)
    r = 25
    H = mask_radial(np.zeros([bs, c, M, N]), r)
    H = np.fft.ifft2(H)
    TS = torch.Tensor(H)
    s = torch.cat((img, TS), 1)
    return s


def test_ffdnet(**args):
    in_ch = argspar.no_of_channel
    # in_ch = 1
    print("No of Channels", in_ch)
    print('Loading data info ...\n')
    path = r'./testingTime/color1024.tiff';
    files_source = glob.glob(path)
    files_source.sort()
    print("File source", len(files_source))
    psnr_test = 0
    ssim_test = 0
    fsim_test = 0
    lpips_test = 0

    if in_ch == 3:
        model_fn = './logs/DFilter/DFilterColor/80/best_model.pth'
        print("model used Path", model_fn)
    else:
        model_fn = './logs/DFilter/DFilterGray/80/best_model.pth'
        print("models is ", model_fn)
    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            model_fn)
    # Create model
    print('Loading model ...\n')
    # net = FFDNet(num_input_channels=in_ch)
    net = FFDNet(num_input_channels=in_ch).to(device)

    # To know number of parameters work in CNN
    total_params = sum(
        param.numel() for param in net.parameters()
    )
    print("Parameters", total_params)
    if args['cuda']:
        state_dict = torch.load(model_fn)
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids)
        model.to(device)
        print("Device", model.to(device))
    else:
        state_dict = torch.load(model_fn, map_location='cpu')
        state_dict = remove_dataparallel_wrapper(state_dict)
        model = net
    model.load_state_dict(state_dict)
    model.eval()
    if args['cuda']:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    for f in files_source:
        if in_ch == 3:
            image = cv2.imread(f)
            imorig = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            imorig = np.expand_dims(image, 0)

        imorig = np.expand_dims(imorig, 0)
        expanded_h = False
        expanded_w = False
        sh_im = imorig.shape

        if sh_im[2] % 2 == 1:
            expanded_h = True
            imorig = np.concatenate((imorig,
                                     imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

        if sh_im[3] % 2 == 1:
            expanded_w = True
            imorig = np.concatenate((imorig,
                                     imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

        imorig = normalize(imorig)
        imorig = torch.FloatTensor(imorig)
        if args['add_noise']:
            noise = torch.FloatTensor(imorig.size()).normal_(mean=0, std=args['noise_sigma'])
            imnoisy = imorig + noise
        else:
            imnoisy = imorig.clone()

        imnoisy = convertFreqImage(imnoisy)

        # Convert to GPU
        with torch.no_grad():
            imorig, imnoisy = Variable(imorig.type(dtype)), \
                              Variable(imnoisy.type(dtype))
            nsigma = Variable(
                torch.FloatTensor([args['noise_sigma']]).type(dtype))
        # print("Device",device)

        imnoisy = imnoisy.to(device)
        nsigma = nsigma.to(device)
        start_time = time.process_time()
        # start_time = time.time()
        outim = torch.clamp(
            imnoisy[:, :1, :, :] - model(imnoisy, nsigma), 0., 1.)
        end_time = time.process_time()  # single threaded
        # end_time = time.time()  # multiple threaded
        t = end_time - start_time

        # To handle width and height different Image size
        if expanded_h:
            imorig = imorig[:, :, :-1, :]
            outim = outim[:, :, :-1, :]
            imnoisy = imnoisy[:, :, :-1, :]

        if expanded_w:
            imorig = imorig[:, :, :, :-1]
            outim = outim[:, :, :, :-1]
            imnoisy = imnoisy[:, :, :, :-1]

        # Evaluate the IQA Measurements(PSNR/SSIM/FSIM)

        psnr = batch_psnr(outim, imorig, 1.)
        ssims = batch_ssim(outim, imorig, 1.)
        if in_ch == 3:
            fsims = batch_fsim(outim, imorig, 1.)
        else:
            fsims = batch_fsimgray(outim, imorig, 1.)

        print("\n%s PSNR: %f" % (f, psnr))
        print("%s SSIM: %f" % (f, ssims))
        print("%s FSIM: %f" % (f, fsims))

        # Evaluate the Time

        print("%s Time: %.2f" % (f, t))


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument('--add_noise', type=str, default="true")
    parser.add_argument("--test_data", type=str, default="./data/gray/Set12",
                        help='path to input image')
    parser.add_argument("--noise_sigma", type=float, default=25,
                        help='noise level used on test set')
    parser.add_argument("--no_gpu", action='store_true',
                        help="run model on CPU")
    parser.add_argument("--no_of_channel", type=int, default=3, help="color for 3 and grayscale for 1")
    argspar = parser.parse_args()
    argspar.noise_sigma /= 255.
    argspar.add_noise = (argspar.add_noise.lower() == 'true')
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()
    print("\n### Testing FFDNet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    test_ffdnet(**vars(argspar))
