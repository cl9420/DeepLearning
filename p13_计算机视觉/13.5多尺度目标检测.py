import torch
from d2l import torch as d2l

img = d2l.plt.imread('../img/3.jpg')
h, w = img.shape[:2]
print(h, w)


def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros(fmap_h, fmap_w)
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)


# display_anchors(fmap_w=4, fmap_h=4, s=[0.2])

display_anchors(fmap_w=1, fmap_h=1, s=[0.9])
d2l.plt.show()
