import argparse
import glob
import os
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def PowerSpectrum(x):
    M, N = x.shape[0], x.shape[1]
    MS = M // 2
    NS = N // 2
    F = np.fft.fftn(x)
    F_magnitude = np.fft.fftshift(F)
    PS = (np.abs(F_magnitude) ** 2)
    Htt = PS[MS,]
    Vtt = PS[:, MS]
    SS = np.sum(np.sum(np.ravel(Htt)) + np.sum(np.ravel(Vtt)))
    Spar = np.sum(np.ravel(PS)) / SS
    # Color (45) , Gray (70)
    Thre = Spar * 45  # Threshold
    K = int(Thre)
    PS[MS - K: MS + K, NS - K: NS + K] = 0
    peaks = PS <= Thre
    iffts = np.fft.ifftshift(peaks).astype(int)
    image_filtered = np.fft.ifft2(iffts)
    return image_filtered


def convertFreqImage(img):  # Convert Frequency Domain
    x = img.to('cpu').detach().numpy().copy()
    bs, c, M, N = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    S = np.zeros((bs, c, M, N), dtype=np.float32)
    for bs in range(bs):
        for ch in range(c):
            S[bs, ch, :, :] = PowerSpectrum(x[bs, ch, :, :])
    TS = torch.Tensor(S)
    s = torch.cat((img, TS), 1)
    return s


def test_ffdnet(**args):
    in_ch = argspar.no_of_channel
    # in_ch = 1
    print("No of Channels", in_ch)
    print('Loading data info ...\n')
    path = r'./data/rgb/Kodak24/*.png';
    files_source = glob.glob(path)
    files_source.sort()
    lpfunc = lpips.LPIPS(net='vgg').cuda()
    print("File source", len(files_source))
    psnr_test = 0
    ssim_test = 0
    fsim_test = 0
    lpips_test = 0

    if in_ch == 3:
        model_fn = './logs/PS/PSColor/80/best_model.pth'
        print("model used Path", model_fn)
    else:
        model_fn = './logs/PS/PSGray/80/best_model.pth'
        print("models is ", model_fn)
    # Absolute path to model file
    model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            model_fn)
    # Create model
    print('Loading model ...\n')
    net = FFDNet(num_input_channels=in_ch)

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

        if in_ch == 1:
            outim = torch.clamp(
                imnoisy[:, :1, :, :].to('cuda') - model(imnoisy.to('cuda'), nsigma.to('cuda')),
                0., 1.)

        else:
            outim = torch.clamp(
                imnoisy[:, :3, :, :].to('cuda') - model(imnoisy.to('cuda'), nsigma.to('cuda')),
                0., 1.)

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

        lpips_value = lpfunc(outim, imorig).item()

        print("\n%s PSNR: %f" % (f, psnr))
        print("%s SSIM: %f" % (f, ssims))
        print("%s FSIM: %f" % (f, fsims))
        print("%s LPIPS: %f" % (f, lpips_value))

        # To Display Noisy Images and Denoised Images
        # For Noisy Images
        # model_image = torch.squeeze(imnoisy[:, :1, :, :])
        # For Denoised Images
        # if in_ch == 1:
        #     model_image = torch.squeeze(outim[:, :1, :, :])
        # else:
        #     model_image = torch.squeeze(outim[:, :3, :, :])
        # model_image = transforms.ToPILImage()(model_image)
        # fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
        # ax0.imshow(model_image).set_cmap("gray")
        # ax0.get_xaxis().set_ticks([])
        # ax0.get_yaxis().set_ticks([])
        # matplotlib.pyplot.box(False)
        # matplotlib.pyplot.show()
        # matplotlib.pyplot.close(fig0)

        psnr_test += psnr
        ssim_test += ssims
        fsim_test += fsims
        lpips_test += lpips_value

    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    fsim_test /= len(files_source)
    lpips_test /= len(files_source)

    print("\nPSNR : %f" % psnr_test)
    print("\nSSIM : %f" % ssim_test)
    print("\nFSIM : %f" % fsim_test)
    print("\nLPIPS : %f" % lpips_test)


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
