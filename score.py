from __future__ import print_function

import os, pdb, sys, glob
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from PIL import Image
import scipy.misc
import inception_score

def l1_mean_dist(x,y):   
    diff = x.astype(float)-y.astype(float)
    return np.sum(np.abs(diff))/np.product(x.shape)

def l2_mean_dist(x,y):   
    diff = x.astype(float)-y.astype(float)
    return np.sqrt(np.sum(diff**2))/np.product(x.shape)

# we need to set GPUno first, otherwise may out of memory
stage = int(sys.argv[1])
gpuNO = sys.argv[2]
model_dir = sys.argv[3]
test_mode = sys.argv[4]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpuNO)

if 1==stage:

    # =====todo: file path configuration ============
    test_result_dir_x = os.path.join(model_dir, test_mode, 'x_target')
    # test_result_dir_x = os.path.join(model_dir, test_mode, 'x')
    test_result_dir_G = os.path.join(model_dir, test_mode, 'G')
    score_path = os.path.join(model_dir, test_mode, 'score_rgb.txt')

    # types = ('*.jpg', '*.png') # the tuple of file types
    # types = ('*.jpg')  # the tuple of file types
    x_files = []
    G_files = []

    # for files in types:
    # x_files.extend(glob.glob(os.path.join(test_result_dir_x, files)))
    # G_files.extend(glob.glob(os.path.join(test_result_dir_G, files)))
    x_files.extend(glob.glob(os.path.join('result/market_g2_final_iter_0425_lambda10/', '*target.jpg')))
    G_files.extend(glob.glob(os.path.join('result/market_g2_final_iter_0425_lambda10/', '*g1out.jpg')))

    x_target_list = []
    for path in x_files:
        x_target_list.append(scipy.misc.imread(path))

    G_list = []
    for path in G_files:
        G_list.append(scipy.misc.imread(path))

    ##################### SSIM ##################
    N = len(x_files)
    ssim_G_x = []
    psnr_G_x = []
    L1_mean_G_x = []
    L2_mean_G_x = []
    # x_0_255 = utils_wgan.unprocess_image(x_fixed, 127.5, 127.5)
    x_0_255 = x_target_list
    # todo: xrange
    for i in range(N):
        # G_gray = rgb2gray((G_list[i]/127.5-1).clip(min=-1,max=1))
        # x_target_gray = rgb2gray((x_target_list[i]/127.5-1).clip(min=-1,max=1))
        # gray image, [0,1]
        # G_gray = rgb2gray((G_list[i]).clip(min=0,max=255))
        # x_target_gray = rgb2gray((x_target_list[i]).clip(min=0,max=255))
        # ssim_G_x.append(ssim(G_gray, x_target_gray, data_range=x_target_gray.max()-x_target_gray.min(), multichannel=False))
        # psnr_G_x.append(psnr(im_true=x_target_gray, im_test=G_gray, data_range=x_target_gray.max()-x_target_gray.min()))
        # color image
        G_gray = G_list[i]
        x_target_gray = x_target_list[i]
        ssim_G_x.append(ssim(G_list[i], x_target_list[i], multichannel=True))
        psnr_G_x.append(psnr(im_true=x_target_gray, im_test=G_gray))
        L1_mean_G_x.append(l1_mean_dist(G_gray, x_target_gray))
        L2_mean_G_x.append(c(G_gray, x_target_gray))
    # pdb.set_trace()
    ssim_G_x_mean = np.mean(ssim_G_x)
    ssim_G_x_std = np.std(ssim_G_x)
    psnr_G_x_mean = np.mean(psnr_G_x)
    psnr_G_x_std = np.std(psnr_G_x)
    L1_G_x_mean = np.mean(L1_mean_G_x)
    L1_G_x_std = np.std(L1_mean_G_x)
    L2_G_x_mean = np.mean(L2_mean_G_x)
    L2_G_x_std = np.std(L2_mean_G_x)
    print('ssim_G_x_mean: %f\n' % ssim_G_x_mean)
    print('ssim_G_x_std: %f\n' % ssim_G_x_std)
    print('psnr_G_x_mean: %f\n' % psnr_G_x_mean)
    print('psnr_G_x_std: %f\n' % psnr_G_x_std)
    print('L1_G_x_mean: %f\n' % L1_G_x_mean)
    print('L1_G_x_std: %f\n' % L1_G_x_std)
    print('L2_G_x_mean: %f\n' % L2_G_x_mean)
    print('L2_G_x_std: %f\n' % L2_G_x_std)

    ##################### Inception score ##################
    IS_G_mean, IS_G_std = inception_score.get_inception_score(G_list)

    print('IS_G_mean: %f\n' % IS_G_mean)
    print('IS_G_std: %f\n' % IS_G_std)

    with open(score_path, 'w')  as f:
        f.write('Image number: %d\n' % N)
        f.write('ssim: %.5f +- %.5f   ' % (ssim_G_x_mean, ssim_G_x_std))
        f.write('IS: %.5f +- %.5f   ' % (IS_G_mean, IS_G_std))
        f.write('psnr: %.5f +- %.5f   ' % (psnr_G_x_mean, psnr_G_x_std))
        f.write('L1: %.5f +- %.5f   ' % (L1_G_x_mean, L1_G_x_std))
        f.write('L2: %.5f +- %.5f' % (L2_G_x_mean, L2_G_x_std))
elif 2==stage:
    test_result_dir_x = os.path.join(model_dir, test_mode, 'x_target')
    test_result_dir_G1 = os.path.join(model_dir, test_mode, 'G1')
    test_result_dir_G2 = os.path.join(model_dir, test_mode, 'G2')
    score_path = os.path.join(model_dir, test_mode, 'score_rgb.txt') #

    # types = ('*.jpg', '*.png') # the tuple of file types
    x_files = []
    G1_files = []
    G2_files = []

    # for files in types:
    #     x_files.extend(glob.glob(os.path.join(test_result_dir_x, files)))
    #     G1_files.extend(glob.glob(os.path.join(test_result_dir_G1, files)))
    #     G2_files.extend(glob.glob(os.path.join(test_result_dir_G2, files)))
    x_files.extend(glob.glob(os.path.join('result/market_g2_final_iter_0425_lambda10/', '*target.jpg')))
    G1_files.extend(glob.glob(os.path.join('result/market_g2_final_iter_0425_lambda10/', '*g1out.jpg')))
    G2_files.extend(glob.glob(os.path.join('result/market_g2_final_iter_0425_lambda10/', '*finalout.jpg')))

    x_target_list = []
    for path in x_files:
        x_target_list.append(scipy.misc.imread(path))
    G1_list = []
    for path in G1_files:
        G1_list.append(scipy.misc.imread(path))
    G2_list = []
    for path in G2_files:
        G2_list.append(scipy.misc.imread(path))

    # # todo: temperoray computation
    # IS_G2_mean, IS_G2_std = inception_score.get_inception_score(G2_list)
    # print('IS_G2_mean: %f\n' % IS_G2_mean)
    # print('IS_G2_std: %f\n' % IS_G2_std)
    # # exit(0)

    ##################### SSIM G1 ##################
    N = len(x_files)
    ssim_G_x = []
    psnr_G_x = []
    L1_mean_G_x = []
    L2_mean_G_x = []
    # todo: xrange
    for i in range(N):
        # G1_gray = rgb2gray((G1_list[i]/127.5-1).clip(min=-1,max=1))
        # x_target_gray = rgb2gray((x_target_list[i]/127.5-1).clip(min=-1,max=1))
        # gray image, [0,1]
        # G1_gray = rgb2gray((G1_list[i]).clip(min=0,max=255))
        # x_target_gray = rgb2gray((x_target_list[i]).clip(min=0,max=255))
        # ssim_G_x.append(ssim(G_gray, x_target_gray, data_range=x_target_gray.max()-x_target_gray.min(), multichannel=False))
        # psnr_G_x.append(psnr(im_true=x_target_gray, im_test=G1_gray, data_range=x_target_gray.max()-x_target_gray.min()))
        # color image
        G1_gray = G1_list[i]
        x_target_gray = x_target_list[i]
        ssim_G_x.append(ssim(G1_list[i], x_target_list[i], multichannel=True))        
        psnr_G_x.append(psnr(im_true=x_target_gray, im_test=G1_gray))        
        L1_mean_G_x.append(l1_mean_dist(G1_gray, x_target_gray))
        L2_mean_G_x.append(l2_mean_dist(G1_gray, x_target_gray))
        
    print(ssim_G_x)
    exit(0)
    ssim_G1_x_mean = np.mean(ssim_G_x)
    ssim_G1_x_std = np.std(ssim_G_x)
    psnr_G1_x_mean = np.mean(psnr_G_x)
    psnr_G1_x_std = np.std(psnr_G_x)
    L1_G1_x_mean = np.mean(L1_mean_G_x)
    L1_G1_x_std = np.std(L1_mean_G_x)
    L2_G1_x_mean = np.mean(L2_mean_G_x)
    L2_G1_x_std = np.std(L2_mean_G_x)
    print('ssim_G1_x_mean: %f\n' % ssim_G1_x_mean)
    print('ssim_G1_x_std: %f\n' % ssim_G1_x_std)
    print('psnr_G1_x_mean: %f\n' % psnr_G1_x_mean)
    print('psnr_G1_x_std: %f\n' % psnr_G1_x_std)
    print('L1_G1_x_mean: %f\n' % L1_G1_x_mean)
    print('L1_G1_x_std: %f\n' % L1_G1_x_std)
    print('L2_G1_x_mean: %f\n' % L2_G1_x_mean)
    print('L2_G1_x_std: %f\n' % L2_G1_x_std)
    ##################### SSIM G2 ##################
    N = len(x_files)
    ssim_G_x = []
    psnr_G_x = []
    L1_mean_G_x = []
    L2_mean_G_x = []
    # x_0_255 = utils_wgan.unprocess_image(x_fixed, 127.5, 127.5)
    # x_0_255 = x_target_list
    for i in range(N):
        # G2_gray = rgb2gray((G2_list[i]/127.5-1).clip(min=-1,max=1))
        # x_target_gray = rgb2gray((x_target_list[i]/127.5-1).clip(min=-1,max=1))
        # gray image, [0,1]
        # G2_gray = rgb2gray((G2_list[i]).clip(min=0,max=255))
        # x_target_gray = rgb2gray((x_target_list[i]).clip(min=0,max=255))
        # ssim_G_x.append(ssim(G_gray, x_target_gray, data_range=x_target_gray.max()-x_target_gray.min(), multichannel=False))
        # psnr_G_x.append(psnr(im_true=x_target_gray, im_test=G2_gray, data_range=x_target_gray.max()-x_target_gray.min()))
        # color image
        G2_gray = G2_list[i]
        x_target_gray = x_target_list[i]
        ssim_G_x.append(ssim(G2_list[i], x_target_list[i], multichannel=True))        
        psnr_G_x.append(psnr(im_true=x_target_gray, im_test=G2_gray))
        L1_mean_G_x.append(l1_mean_dist(G2_gray, x_target_gray))
        L2_mean_G_x.append(l2_mean_dist(G2_gray, x_target_gray))
    # pdb.set_trace()
    ssim_G2_x_mean = np.mean(ssim_G_x)
    ssim_G2_x_std = np.std(ssim_G_x)
    psnr_G2_x_mean = np.mean(psnr_G_x)
    psnr_G2_x_std = np.std(psnr_G_x)
    L1_G2_x_mean = np.mean(L1_mean_G_x)
    L1_G2_x_std = np.std(L1_mean_G_x)
    L2_G2_x_mean = np.mean(L2_mean_G_x)
    L2_G2_x_std = np.std(L2_mean_G_x)
    print('ssim_G2_x_mean: %f\n' % ssim_G2_x_mean)
    print('ssim_G2_x_std: %f\n' % ssim_G2_x_std)
    print('psnr_G2_x_mean: %f\n' % psnr_G2_x_mean)
    print('psnr_G2_x_std: %f\n' % psnr_G2_x_std)
    print('L1_G2_x_mean: %f\n' % L1_G2_x_mean)
    print('L1_G2_x_std: %f\n' % L1_G2_x_std)
    print('L2_G2_x_mean: %f\n' % L2_G2_x_mean)
    print('L2_G2_x_std: %f\n' % L2_G2_x_std)

    ##################### Inception score ##################

    IS_G1_mean, IS_G1_std = inception_score.get_inception_score(G1_list)
    print('IS_G1_mean: %f\n' % IS_G1_mean)
    print('IS_G1_std: %f\n' % IS_G1_std)
    IS_G2_mean, IS_G2_std = inception_score.get_inception_score(G2_list)
    print('IS_G2_mean: %f\n' % IS_G2_mean)
    print('IS_G2_std: %f\n' % IS_G2_std)

    with open(score_path, 'w')  as f:
        f.write('N: %d   ' % N)
        f.write('ssimG1: %.5f +- %.5f   ' % (ssim_G1_x_mean, ssim_G1_x_std))
        f.write('ISG1: %.5f +- %.5f   ' % (IS_G1_mean, IS_G1_std))
        f.write('psnrG1: %.5f +- %.5f   ' % (psnr_G1_x_mean, psnr_G1_x_std))
        f.write('L1G1: %.5f +- %.5f   ' % (L1_G1_x_mean, L1_G1_x_std))
        f.write('L2G1: %.5f +- %.5f   ' % (L2_G1_x_mean, L2_G1_x_std))
        f.write('ssimG2: %.5f +- %.5f   ' % (ssim_G2_x_mean, ssim_G2_x_std))
        f.write('ISG2: %.5f +- %.5f   ' % (IS_G2_mean, IS_G2_std))
        f.write('psnrG2: %.5f +- %.5f   ' % (psnr_G2_x_mean, psnr_G2_x_std))
        f.write('L1G2: %.5f +- %.5f   ' % (L1_G2_x_mean, L1_G2_x_std))
        f.write('L2G2: %.5f +- %.5f' % (L2_G2_x_mean, L2_G2_x_std))

