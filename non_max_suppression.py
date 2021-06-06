import numpy as np

#overlap Threshold smaller value smaller overlap area, vice versa.
def non_max_supperssion(bboxes, overlapThreshold=0.4):
    if len(bboxes) == 0:
        return []

    if bboxes.dtype.kind == "i":
        bboxes = bboxes.astype("float")

    pickBox = []
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    area = (x2-x1+1)*(y2-y1+1) #non-zero is required
    #sort the bottom right y coordinate in ascending order.
    idx = np.argsort(y2)

    while len(idx)>0:
        lastIdx = len(idx)-1
        i = idx[lastIdx]
        pickBox.append(i)

        xx1 = np.maximum(x1[i], x1[idx[:lastIdx]])
        yy1 = np.maximum(y1[i], y1[idx[:lastIdx]])
        xx2 = np.minimum(x2[i], x2[idx[:lastIdx]])
        yy2 = np.minimum(y2[i], y2[idx[:lastIdx]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        iou = (w * h) / area[idx[:lastIdx]]
        idx = np.delete(idx, np.concatenate(([lastIdx],
                                             np.where(iou>overlapThreshold)[0])))

    return bboxes[pickBox].astype("float")