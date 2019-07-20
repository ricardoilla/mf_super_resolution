# This code was taken from : https://github.com/RobertGawron/supper-resolution
# (Robert Gawron)

import sys
import os
import math
from PIL import Image
import numpy

psf =    [  2.66971863e-03,   5.36227322e-02,   1.45761699e-01,   5.36227322e-02,    2.66971863e-03,
                5.36227322e-02,   1.07704137e+00,   2.92770198e+00,   1.07704137e+00,    5.36227322e-02,
                1.45761699e-01,   2.92770198e+00,   7.95831909e+00,   2.92770198e+00,    1.45761699e-01,
                5.36227322e-02,   1.07704137e+00,   2.92770198e+00,   1.07704137e+00,    5.36227322e-02,
                2.66971863e-03,   5.36227322e-02,   1.45761699e-01,   5.36227322e-02,    2.66971863e-03 ],


## helper function for matrix padding
def do_padding(matrix, pwidth, pval=0):
    _pad = numpy.ones(pwidth) * pval
    return numpy.apply_along_axis(lambda col: numpy.concatenate((_pad, col, _pad)), 0,
                                  numpy.apply_along_axis(lambda row: numpy.concatenate((_pad, row, _pad)), 1, matrix))


class Camera:
    def __init__(self, hps):
        print('Creating Camera Model')

        # hps converted to 1D list
        self.hps = numpy.array(hps).reshape(-1).tolist()
        self.hps = self.hps / numpy.sum(numpy.abs(self.hps))  # do normalization

        # size: pixels north of center
        self.size = int((numpy.sqrt(len(self.hps)) - 1) / 2)

        # psf: hps converted to 2D array
        self.psf = numpy.array(self.hps).reshape((2 * self.size + 1, 2 * self.size + 1))

        # square of PSF = BP
        self.psf2 = self.psf * self.psf

        # px_area : list of coordinates for area around center (=(0,0))
        mg = numpy.mgrid[-self.size:self.size + 1, -self.size:self.size + 1]
        self.pxarea = zip(list(mg[0].reshape(-1).tolist()), list(mg[1].reshape(-1).tolist()))

    def take(self, image, offset, scale):

        # convert to numpy array
        odata = numpy.asarray(image).astype(numpy.int32)

        # apply offset to HR image
        odata[:, :, 0] = self.doOffset(odata[:, :, 0], offset)
        odata[:, :, 1] = self.doOffset(odata[:, :, 1], offset)
        odata[:, :, 2] = self.doOffset(odata[:, :, 2], offset)

        # filter with the PSF (one color at the time)
        odata[:, :, 0] = self.Convolve(odata[:, :, 0])
        odata[:, :, 1] = self.Convolve(odata[:, :, 1])
        odata[:, :, 2] = self.Convolve(odata[:, :, 2])

        # convert back to image format
        photo = Image.fromarray(numpy.uint8(odata))

        # apply scale factor
        new_img_sz = int(image.size[0] * scale), int(image.size[1] * scale)
        return photo.resize(new_img_sz, Image.ANTIALIAS)

    def doOffset(self, data, offset, val=255):
        # apply offset (via slicing, vertical and horizontal separately)
        if offset[1] > 0: data = numpy.concatenate(
            (numpy.ones((offset[1], data.shape[1])) * val, data[0:-offset[1], :]), axis=0)
        if offset[1] < 0: data = numpy.concatenate(
            (data[-offset[1]:, :], numpy.ones((-offset[1], data.shape[1])) * val), axis=0)
        if offset[0] > 0: data = numpy.concatenate(
            (numpy.ones((data.shape[0], offset[0])) * val, data[:, 0:-offset[0]]), axis=1)
        if offset[0] < 0: data = numpy.concatenate(
            (data[:, -offset[0]:], numpy.ones((data.shape[0], -offset[0])) * val), axis=1)
        return data

    """
    def Convolve(self, data):
        ### FFT-iFFT approach : does not converge to solution
        fft  = numpy.fft.fft2(data) * numpy.fft.fft2(self.psf, data.shape)
        conv = numpy.fft.ifft2(fft).real
        return conv
    """

    """
    def Convolve(self, data):
        ### nested for-loops implementation : very slow !!!
        conv = data
        w = self.size
        for x in range(w, data.shape[0]-w):
            for y in range(w, data.shape[1]-w):
                conv[x,y] = numpy.sum(data[x-w:x+w+1,y-w:y+w+1] * self.psf)
        return conv
    """

    def Convolve(self, data):
        ### python magic implementation
        w = 2 * self.size
        # need some (zero) padding first
        data = do_padding(data, self.size)
        # now stack row shifts
        b = data[:, 0:-w]
        for r in range(1, w): b = numpy.dstack((b, data[:, r:r - w]))
        b = numpy.dstack((b, data[:, w:]))
        data = b
        # next stack col shifts
        b = data[0:-w, :]
        for c in range(1, w): b = numpy.dstack((b, data[c:c - w, :]))
        b = numpy.dstack((b, data[w:, :]))
        data = b
        # now apply filtering
        conv = numpy.sum(data * numpy.tile(self.hps, (data.shape[0], data.shape[1], 1)), axis=2)

        return conv

    def Convolve2(self, data):
        ### python magic implementation
        w = 2 * self.size
        # need some (zero) padding first
        data = do_padding(data, self.size)
        # now stack row shifts
        b = data[:, 0:-w]
        for r in range(1, w): b = numpy.dstack((b, data[:, r:r - w]))
        b = numpy.dstack((b, data[:, w:]))
        data = b
        # next stack col shifts
        b = data[0:-w, :]
        for c in range(1, w): b = numpy.dstack((b, data[c:c - w, :]))
        b = numpy.dstack((b, data[w:, :]))
        data = b
        # now apply filtering
        conv = numpy.sum(data * numpy.tile((self.hps * self.hps), (data.shape[0], data.shape[1], 1)), axis=2)

        return conv



def clipto_0(val):
    return val if val > 0 else 0


def clipto_255(val):
    return val if val < 255 else 255


def clip(val):
    return clipto_0(clipto_255(val))


cliparray = numpy.frompyfunc(clip, 1, 1)


# upsample an array with zeros
def upsample(arr, n):
    z = numpy.zeros(len(arr))  # upsample with values
    for i in range(int(int(n - 1) / 2)):  # TODO
        arr = numpy.dstack((z, arr))
    for i in range(int(int(n) / 2)):  # TODO
        arr = numpy.dstack((arr, z))
    return arr.reshape((1, -1))[0]


def SRRestore(camera, origImg, samples, upscale, iter):
    error = 0

    high_res_new = numpy.asarray(origImg).astype(numpy.float32)

    # for every LR with known pixel-offset
    for (offset, captured) in samples:
        (dx, dy) = offset

        # make LR of HR given current pixel-offset
        simulated = camera.take(origImg, offset, 1.0 / upscale)

        # convert captured and simulated to numpy arrays (mind the data type!)
        cap_arr = numpy.asarray(captured).astype(numpy.float32)
        sim_arr = numpy.asarray(simulated).astype(numpy.float32)

        # get delta-image/array: captured - simulated
        delta = (cap_arr - sim_arr) / len(samples)

        # Sum of Absolute Difference Error
        error += numpy.sum(numpy.abs(delta))

        # upsample delta to HR size (with zeros)
        delta_hr_R = numpy.apply_along_axis(
            lambda row: upsample(row, upscale),
            1,
            numpy.apply_along_axis(
                lambda col: upsample(col, upscale),
                0,
                delta[:, :, 0]))

        delta_hr_G = numpy.apply_along_axis(
            lambda row: upsample(row, upscale),
            1,
            numpy.apply_along_axis(
                lambda col: upsample(col, upscale),
                0,
                delta[:, :, 1]))

        delta_hr_B = numpy.apply_along_axis(
            lambda row: upsample(row, upscale),
            1,
            numpy.apply_along_axis(
                lambda col: upsample(col, upscale),
                0, delta[:, :, 2]))

        # apply the offset to the delta
        delta_hr_R = camera.doOffset(delta_hr_R, (-dx, -dy))
        delta_hr_G = camera.doOffset(delta_hr_G, (-dx, -dy))
        delta_hr_B = camera.doOffset(delta_hr_B, (-dx, -dy))

        # Blur the (upsampled) delta with PSF
        delta_hr_R = camera.Convolve(delta_hr_R)
        delta_hr_G = camera.Convolve(delta_hr_G)
        delta_hr_B = camera.Convolve(delta_hr_B)

        # and update high_res image with filter result
        high_res_new += numpy.dstack((delta_hr_R,
                                      delta_hr_G,
                                      delta_hr_B))

    # normalize image array again (0-255)
    high_res_new = cliparray(high_res_new)

    return Image.fromarray(numpy.uint8(high_res_new)), error


def loadSamples(directory):
    samples = []

    for sampleFileName in (os.listdir(directory)):
        print(sampleFileName)
        sampleExtension = sampleFileName[-4:]
        if sampleExtension != '.tif':
            # print("" % (fileExtension))
            continue

        sample = Image.open(directory + '/' + sampleFileName)
        if not samples:
            samples.append(((0, 0), sample))
        else:
            (x, y) = 0, 0
            samples.append(((x, y), sample))
    return samples


def SR_main(sampleDirectory, scale, ite, output_name):
    print("Estimate Motion Between Sample And Original Image")
    samples = loadSamples(sampleDirectory)
    print("samples loaded", samples)
    # print ("Restore SR Image")
    camera = Camera(psf)

    scale = scale
    origSizeX = samples[0][1].size[1] * scale
    origSizeY = samples[0][1].size[0] * scale
    origImage = numpy.zeros([origSizeX, origSizeY, 3]).astype(numpy.float32)
    print("Size Of Estimated Original: %dx%d" % (origSizeX, origSizeY))

    for ((dx, dy), sample) in samples:
        sampleOrigSize = sample.resize((origSizeX, origSizeY), Image.ANTIALIAS)
        sampleAsArr = numpy.asarray(sampleOrigSize)

    origImage = origImage / len(samples)  # take average value
    origImage = Image.fromarray(numpy.uint8(origImage))

    # TODO move this to a separate class
    for i in range(ite):
        origImage, estimDiff = SRRestore(camera, origImage, samples, scale, i)
        # estimDiff = 5
        estimDiff /= float(origSizeX * origSizeY)
        print('%2d: estimation error: %3f' % (i, estimDiff))

    origImage.save(os.path.join(sampleDirectory, output_name))


