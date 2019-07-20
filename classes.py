__author__ = 'RicardoIlla'

from functions import *
import glob, time
from deblur import *


class SRStep:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path


class Filter(SRStep):
    def __init__(self, input_path, output_path):
        SRStep.__init__(self, input_path, output_path)

    def run(self):
        filenames = glob.glob(self.input_path)
        filenames.sort()
        # Upload all images
        raw_images = [cv2.imread(img) for img in filenames]
        sharps = []
        for x in range(len(raw_images)):
            value = variance_of_laplacian(raw_images[x])
            print('File {} has blur:{}'.format(filenames[x], value))
            sharps.append(value)

        mean = np.mean(sharps)
        print('Mean: ', mean)
        for x in range(len(raw_images)):
            if sharps[x] - mean < 0:
                print('Discard: {}'.format(filenames[x]))
            else:
                im = cv2.imread(filenames[x])
                print('Saving filtered as {}'.format(self.output_path + filenames[x][6:]))
                cv2.imwrite(self.output_path + filenames[x][6:], im)


class NLMDenoise(SRStep):
    def __init__(self, input_path, output_path):
        SRStep.__init__(self, input_path, output_path)
        self.PATCH_SIZE = 7
        self.PATCH_DISTANCE = 9
        self.H = 0.08
        self.MULTICHANNEL = True
        self.FAST_MODE = True

    def run(self):
        filenames = glob.glob(self.input_path)
        filenames.sort()
        # Upload all images
        images = [cv2.imread(img) for img in filenames]

        for x in range(len(images)):
            denoise = denoise_nl_means(images[x], self.PATCH_SIZE, self.PATCH_DISTANCE, self.H, multichannel=self.MULTICHANNEL)
            cv2.imwrite(self.output_path + str(x) + '.tif', denoise)


class Upsample(SRStep):
    def __init__(self, input_path, output_path):
        SRStep.__init__(self, input_path, output_path)

    def run(self):
        filenames = glob.glob(self.input_path)
        filenames.sort()
        # Upload all images
        images = [cv2.imread(img) for img in filenames]
        for x in range(len(images)):
            new_img = bicubic_upsample(images[x], ratio=2)
            cv2.imwrite(self.output_path + str(x) + '.tif', new_img)
            print('New image upsampled:')
            print(new_img.shape[:2])


class Align(SRStep):
    def __init__(self, input_path, output_path):
        SRStep.__init__(self, input_path, output_path)

    def run(self):
        # Upload all upsampled images
        filenames = glob.glob(self.input_path)
        filenames.sort()
        data = [cv2.imread(img) for img in filenames]
        cv2.imwrite(self.output_path+'0.tif', data[0])
        print('Saving aligned as {}'.format(self.output_path+'0.tif'))
        result = data[0]
        for x in range(1, len(data) - 1):
            result = Perspective_warping(result, data[x])
            print('Saving aligned as {}'.format(self.output_path + str(x) + '.tif'))
            cv2.imwrite(self.output_path + str(x) + '.tif', result)


class Merge(SRStep):
    def __init__(self, input_path, output_path):
        SRStep.__init__(self, input_path, output_path)

    def run(self):
        filenames = glob.glob(self.input_path)
        filenames.sort()
        # Upload all images
        images = [cv2.imread(img) for img in filenames]
        # print(images)
        total = cv2.addWeighted(images[0], 0.5, images[1], 0.5, 0)

        # More focus -> More weight
        for x in range(2, len(images)):
            l1 = variance_of_laplacian(total)
            l2 = variance_of_laplacian(images[x])
            total = cv2.addWeighted(total, (l1 / (l1 + l2)), images[x], (l2 / (l1 + l2)), 0)
        print('Saving Merge as {}'.format(self.output_path))
        cv2.imwrite(self.output_path, total)


class Deblur(SRStep):
    def __init__(self, input_path, output_path):
        SRStep.__init__(self, input_path, output_path)

    def run(self):
        startTime = time.time()
        SR_main(self.input_path, 1, 30, output_name=self.output_path)
        print('Total elapsed time: %s mins' % str((time.time() - startTime) / 60))



