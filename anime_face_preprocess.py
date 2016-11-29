import os
import sys
import random
import numpy as np
import scipy.misc
import cv2

from tensorflow.contrib.learn.python.learn.datasets import base


_MY_DIR = os.path.dirname(os.path.realpath(__file__))
_SRC_DATA_FOLDER = "thumb"
_DEST_DATA_FOLDER = "thumb_face"

_CASCADE_FILE_SOURCE_URL = ("https://raw.githubusercontent.com/nagadomi/"
                            "lbpcascade_animeface/master/")
_CASCADE_FILE_FILE = "lbpcascade_animeface.xml"


class FaceDetector(object):

    def __init__(self):
        cascade_file = base.maybe_download(
            _CASCADE_FILE_FILE, _MY_DIR,
            _CASCADE_FILE_SOURCE_URL + _CASCADE_FILE_FILE)
        cascade_classifier = cv2.CascadeClassifier(cascade_file)
        self._cascade_file = cascade_file
        self._cascade_classifier = cascade_classifier

    def run(self, image):
        # image has the format H x W x RGB
        if not os.path.isfile(self._cascade_file):
            raise RuntimeError("%s: not found" % cascade_file)

        image = image[:,:,::-1]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        return self._cascade_classifier.detectMultiScale(
            gray,
            # detector options
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (24, 24))


def _resolve_file_dirs(data_root_dir):
    data_dir = os.path.join(data_root_dir, _SRC_DATA_FOLDER)
    ans = []
    for d in os.listdir(data_dir):
        p = os.path.normpath(os.path.join(data_dir, d))
        if os.path.isdir(p):
            ans.append(p)
    return ans


def _process_image(im, face_rect):
    x, y, w, h = face_rect
    x, y = max(x, 0), max(y, 0)
    if len(im.shape) == 2:
        im = im[y:y+h, x:x+w]
        im = np.tile(np.expand_dims(im, 3), (1, 1, 3))
    else:
        im = im[y:y+h, x:x+w, 0:3]

    h, w = im.shape[:2]
    a = min(h, w)
    y, x = h // 2 - a // 2, w // 2 - a // 2
    im = im[y:y+a, x:x+a, 0:3]
    return scipy.misc.imresize(im, [64, 64]) / 255.0


def _main(data_root_dir):
    face_detector = FaceDetector()
    file_dirs = _resolve_file_dirs(data_root_dir)
    dest_dir = os.path.join(data_root_dir, _DEST_DATA_FOLDER)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_dir in file_dirs:
        for file_name in os.listdir(file_dir):
            if not file_name.endswith(".png"):
                continue
            im = scipy.misc.imread(os.path.join(file_dir, file_name))
            face_rects = face_detector.run(im)
            for i, face_rect in enumerate(face_rects):
                save_file_path = os.path.join(
                    dest_dir, "{:s}_{:d}.png".format(file_name, i))
                try:
                    face_im = _process_image(im, face_rect)
                    scipy.misc.toimage(face_im, cmin=0.0, cmax=1.0).save(
                        save_file_path)
                except Exception as e:
                    print "Error: {:s}".format(e)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        data_root_dir = os.path.join(_MY_DIR, "../anime_face")
    else:
        data_root_dir = sys.argv[1]
    _main(data_root_dir)
