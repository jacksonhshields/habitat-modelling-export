import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from habitat_modelling.utils.display_utils import colour_bathy_patch
import cv2
import neptune
import os


def fig2pil(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)

    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())

def sample_iae(real_imgs, i2i, epoch, save_image_dir, img_postprocessor, num_imgs=1, text=True, neptune_exp=None):
    for n in range(num_imgs):
        # Do images
        rimg = np.array(img_postprocessor(real_imgs[n, ::]))
        geni2i = np.array(img_postprocessor(i2i[n, ::]))
        img_shape2d = rimg.shape[:2]

        # Check for bw images and if they are convert them to colour
        if len(rimg.shape) == 2:
            rimg = cv2.cvtColor(rimg, cv2.COLOR_GRAY2RGB)
        if len(geni2i.shape) == 2:
            geni2i = cv2.cvtColor(geni2i, cv2.COLOR_GRAY2RGB)

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.005 * img_shape2d[0]
            font_position = (int(img_shape2d[0] / 2), int(img_shape2d[1] - font_size) - 2)
            font_thickness = int(img_shape2d[0] / 64)
            cv2.putText(rimg, 'r', font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(geni2i, 'g', font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)

        combined = np.vstack((rimg, geni2i))

        combined_p = Image.fromarray(combined)
        combined_p.save(os.path.join(os.path.join(save_image_dir, 'iae_e%d_s%d.png'% (epoch, n))))

        if neptune_exp:
            neptune_exp.send_image("iae_s%d" % (n), combined_p)


def sample_bae(real_bathy, depth_batch, b2b, epoch, save_image_dir, num_imgs=1, text=True, neptune_exp=None):
    for n in range(num_imgs):
        # Just for resizing
        img_shape2d = (64, 64)

        rbat = colour_bathy_patch(np.squeeze(real_bathy[n, ::], axis=0), colour_map='jet')
        genb2b = colour_bathy_patch(np.squeeze(b2b[n, ::], axis=0), colour_map='jet')

        rbat = cv2.resize(rbat, img_shape2d)
        genb2b = cv2.resize(genb2b, img_shape2d)
        depth = depth_batch[n]

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.005 * img_shape2d[0]
            font_position = (int(img_shape2d[0] / 2), int(img_shape2d[1] - font_size) - 2)
            font_thickness = int(img_shape2d[0] / 64)
            cv2.putText(rbat, 'd %.2f'%depth, font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(genb2b, 'g', font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
        combined = np.vstack((rbat, genb2b))

        combined_p = Image.fromarray(combined)
        combined_p.save(os.path.join(os.path.join(save_image_dir, 'bae_e%d_s%d.png'% (epoch, n))))

        if neptune_exp:
            neptune_exp.send_image("xae_s%d" % (n), combined_p)

def sample_back_ae(real_bathy, b2b, epoch, save_image_dir, num_imgs=1, text=True, neptune_exp=None):
    for n in range(num_imgs):
        # Just for resizing
        img_shape2d = (64, 64)

        rbat = colour_bathy_patch(np.squeeze(real_bathy[n, ::], axis=0), colour_map='jet')
        genb2b = colour_bathy_patch(np.squeeze(b2b[n, ::], axis=0), colour_map='jet')

        rbat = cv2.resize(rbat, img_shape2d)
        genb2b = cv2.resize(genb2b, img_shape2d)

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.005 * img_shape2d[0]
            font_position = (int(img_shape2d[0] / 2), int(img_shape2d[1] - font_size) - 2)
            font_thickness = int(img_shape2d[0] / 64)
            cv2.putText(rbat, 'b', font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(genb2b, 'g', font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
        combined = np.vstack((rbat, genb2b))

        combined_p = Image.fromarray(combined)
        combined_p.save(os.path.join(os.path.join(save_image_dir, 'backscatter_e%d_s%d.png'% (epoch, n))))

        if neptune_exp:
            neptune_exp.send_image("xae_s%d" % (n), combined_p)

def sample_bathy_back_ae(input_patch, b2b, epoch, save_image_dir, num_imgs=1, text=True, neptune_exp=None):
    for n in range(num_imgs):
        # Just for resizing
        img_shape2d = (64, 64)

        rbat = colour_bathy_patch(input_patch[n,0, ::], colour_map='jet')
        rback = colour_bathy_patch(input_patch[n,1, ::], colour_map='jet')
        genbathy = colour_bathy_patch(b2b[n,0, ::], colour_map='jet')
        genback = colour_bathy_patch(b2b[n,1, ::], colour_map='jet')

        rbat = cv2.resize(rbat, img_shape2d)
        rback = cv2.resize(rback, img_shape2d)
        genbathy = cv2.resize(genbathy, img_shape2d)
        genback = cv2.resize(genback, img_shape2d)

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.005 * img_shape2d[0]
            font_position = (int(img_shape2d[0] / 2), int(img_shape2d[1] - font_size) - 2)
            font_thickness = int(img_shape2d[0] / 64)
            cv2.putText(rbat, 'bm', font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(rback, 'bs', font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(genbathy, 'gm', font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(genback, 'gs', font_position, font, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)

        reals = np.hstack((rbat, rback))
        gens = np.hstack((genbathy, genback))
        combined = np.vstack((reals, gens))

        combined_p = Image.fromarray(combined)
        combined_p.save(os.path.join(os.path.join(save_image_dir, 'bathy_backscatter_ae_e%d_s%d.png'% (epoch, n))))

        if neptune_exp:
            neptune_exp.send_image("xae_s%d" % (n), combined_p)