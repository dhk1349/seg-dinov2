from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

coco = COCO('/mnt/2tb/mscoco/annotations/instances_val2017.json')

img = coco.imgs[210394]
img_dir = '/mnt/2tb/mscoco/val2017' 

image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
plt.imshow(image)
cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids)
coco.showAnns(anns)
plt.show()
