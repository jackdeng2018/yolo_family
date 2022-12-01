import torch
import torchvision
from torch import Tensor


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    :param boxes1:  Tensor[N, 4]
    :param boxes2: Tensor[M, 4]
    :return: iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2

    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    # left-top  [N,M,2] # N中一个和M个比较； 所以由N，M 个
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # right-bottom [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter)


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    nms
    :param boxes: [N,4]
    :param scores: [N]
    :param iou_threshold: 0.7
    :return:
    """
    # keep indexes in the boxes
    keep = []
    # scores sort from small to large
    idxs = scores.argsort()
    # loop to idxs has 0 element
    while idxs.numel() > 0:
        # get largest box index
        max_score_index = idxs[-1]
        # [1, 4]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        # only one box left
        if idxs.size(0) == -1:
            break
        idxs = idxs[:-1]
        # compute iou: other boxes with max_score_box
        other_boxes = boxes[idxs]
        ious = box_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]
    keep = idxs.new(keep)
    return keep


def nms(boxes, scores, overlap=0.7, top_k=200):
    pass


def softnms():
    pass


# 测试NMS的代码；
def test1():
    pro_boxes = torch.load("Boxes_forNMS_test.pt")

    one_boxes = pro_boxes["boxes"][0]  # [1000, 4]
    one_scores = pro_boxes["scores"][0]  # [1000]

    keep = nms(one_boxes, one_scores, 0.7)
    print(keep)
    print(keep.shape)

    ## pytorch nms 接口
    import torchvision
    keep_1 = torchvision.ops.nms(one_boxes, one_scores, 0.7)
    print("#"*20)
    print(keep_1)
    print(keep_1.shape)
    print(keep_1==keep)


if __name__ == "__main__":
    test1()
