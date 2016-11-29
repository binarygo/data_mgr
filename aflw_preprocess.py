import os
import sys
import random
import sqlite3
import numpy as np
import scipy.misc


_MY_DIR = os.path.dirname(os.path.realpath(__file__))
_SRC_DATA_FOLDER = "flickr"
_DEST_DATA_FOLDER = "flickr_face_1"

_SQL_FACE_METADATA = r"""
SELECT
  f.file_id AS file_id,
  fr.x as x,
  fr.y as y,
  fr.w as w,
  fr.h as h
FROM
  facerect AS fr
  JOIN faces AS f ON fr.face_id = f.face_id
WHERE
  fr.annot_type_id = 1
"""


def _read_face_metadata(data_root_dir):
    aflw_sqlite_file = os.path.join(data_root_dir, "aflw.sqlite")
    with sqlite3.connect(aflw_sqlite_file) as conn:
        c = conn.cursor()
        c.execute(_SQL_FACE_METADATA)
        data = c.fetchall()
        face_metadata = {}
        for file_id, x, y, w, h in data:
            face_metadata.setdefault(file_id, []).append((x, y, w, h))
        # face_metadata has the format file_id -> [(x, y, w, h), ...]
        return face_metadata


def _resolve_file_dirs(data_root_dir):
    data_dir = os.path.join(data_root_dir, _SRC_DATA_FOLDER)
    if isinstance(data_dir, basestring):
        ans = []
        for d in os.listdir(data_dir):
            p = os.path.normpath(os.path.join(data_dir, d))
            if os.path.isdir(p):
                ans.append(p)
        return ans
    return file_dirs


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
    return scipy.misc.imresize(im, [128, 128]) / 255.0


def _main(data_root_dir):
    face_metadata = _read_face_metadata(data_root_dir)
    file_dirs = _resolve_file_dirs(data_root_dir)
    dest_dir = os.path.join(data_root_dir, _DEST_DATA_FOLDER)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_dir in file_dirs:
        for file_name in os.listdir(file_dir):
            if not (file_name.endswith(".jpg") and
                    file_name in face_metadata):
                continue
            im = scipy.misc.imread(os.path.join(file_dir, file_name))
            face_rects = face_metadata[file_name]
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
        data_root_dir = os.path.join(_MY_DIR, "../aflw/aflw/data")
    else:
        data_root_dir = sys.argv[1]
    _main(data_root_dir)
