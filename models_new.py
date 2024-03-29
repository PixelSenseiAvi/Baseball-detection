from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forwardlayer(self, x, targets=None, img_dim=None):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss


def upsample(stride):
    return Upsample(scale_factor=int(stride), mode="nearest")


def yolo(mask, anchors, classes, num,jitter, ignore_thresh, truth_thresh, random, height):
    anchor_idxs = mask
    anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in anchor_idxs]
    num_classes = classes
    img_size = height

    yolo_layer = YOLOLayer(anchors, num_classes, img_size)
    #net_layers.append(yolo_layer)
    return yolo_layer

# class Block(nn.Module):
#
#
#     def __init__(self, input_channels, multiplier = 2):
#         super(Block, self).__init__()
#         self.activation = nn.LeakyReLU(0.1)
#         self.bn1 = nn.BatchNorm2d(input_channels, 0.9, 1e-5)
#         self.bn2 = nn.BatchNorm2d(input_channels/multiplier, 0.9, 1e-5)
#
#         self.conv1 = nn.Conv2d(input_channels, input_channels/multiplier, 1, 1, 1)
#         self.conv2 = nn.Conv2d(input_channels/multiplier, input_channels, 3, 1, 1)
#         self.route2
#
#     def forward(self, input):
#         x = self.conv1(input)
#         self.route2 = x
#         x = self.bn1(x)
#         x = self.activation(x)
#         x = self.conv2(input)
#         x = self.bn2(x)
#         x = self.activation(x)
#
#         return x + input
#
#     def return_route():
#         return self.route2


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, channels =3):
        super(Darknet, self).__init__()

        #hyperparams
        self.batch=1
        self.subdivisions=1
        self.width=416
        self.height=416
        self.channels=3
        self.momentum=0.9
        self.decay=0.0005
        self.angle=0
        self.saturation = 1.5
        self.exposure = 1.5
        self.hue=.1

        self.learning_rate=0.001
        self.burn_in=1000
        self.max_batches = 500200
        self.steps=400000,450000
        self.policy=self.steps
        self.scales=.1,.1
        self.bn = 1

        self.layers = []

        self.activation = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16, 0.9, 1e-5)
        self.mp1 = nn.MaxPool2d(16, stride=2, padding=int((2 - 1) // 2))

        self.conv2 = nn.Conv2d(16, 32, 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(32, 0.9, 1e-5)
        self.mp2 = nn.MaxPool2d(32, stride = 2, padding=int((2 - 1) // 2))

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64, 0.9, 1e-5)
        self.mp3 = nn.MaxPool2d(64, stride = 2, padding=int((2 - 1) // 2))

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128, 0.9, 1e-5)
        self.mp4 = nn.MaxPool2d(128, stride = 2, padding=int((2 - 1) // 2))

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256, 0.9, 1e-5)
        self.mp5 = nn.MaxPool2d(256, stride = 2, padding=int((2 - 1) // 2))

        self.conv6 = nn.Conv2d(256, 52, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(52, 0.9, 1e-5)
        self.mp6 = nn.MaxPool2d(52, stride = 1, padding=int((2 - 1) // 2))

        self.conv7 = nn.Conv2d(52, 1024, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(1024, 0.9, 1e-5)

        #######
        self.conv8 = nn.Conv2d(1024, 256, 1, 1, 1)
        self.bn8 = nn.BatchNorm2d(256, 0.9, 1e-5)

        self.conv9 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn9 = nn.BatchNorm2d(512, 0.9, 1e-5)

        self.conv10 = nn.Conv2d(256, 18, 1, 1, 1)
        self.bn10 = nn.BatchNorm2d(18, 0.9, 1e-5)
        # #self.upsample = upsample(2)
        #
        #
        # self.conv11 = nn.Conv2d(256, 256,1,1,1 )
        # self.bn11 = nn.BatchNorm2d(256, 0.9, 1e-5)
        # self.conv12 = nn.Conv2d(256, 512,1,1,1 )
        # self.bn12 = nn.BatchNorm2d(512, 0.9, 1e-5)
        # self.conv13 = nn.Conv2d(512, 18, 1, 1, 1)
        #
        # self.conv14 = nn.Conv2d(512, 128, 1, 1, 1)
        # self.bn14 = nn.BatchNorm2d(128, 0.9, 1e-5)
        # self.conv15 = nn.Conv2d(128, 128, 1, 1, 1)
        # self.bn15 = nn.BatchNorm2d(128, 0.9, 1e-5)

        # self.conv16 = nn.Conv2d(128,18, 1, 1, 1)

        #yolo1
        mask1 = [3,4,5]
        anchors1 = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
        classes1=1
        num1=6
        jitter1=.3
        ignore_thresh1 = .7
        truth_thresh1 = 1
        random1=1
        self.yolo1 = yolo(mask1, anchors1, classes1, num1, jitter1, ignore_thresh1, truth_thresh1, random1, self.height)

        # check for 18 or 1, should be 18 most probably
        self.conv11 = nn.Conv2d(18, 128, 1, 1, 1)
        self.bn11 = nn.BatchNorm2d(128, 0.9, 1e-5)

        self.upsample1 = upsample(2)

        #22
        self.conv12 = nn.Conv2d(128, 256, 1, 1, 1)
        self.bn12 = nn.BatchNorm2d(256, 0.9, 1e-5)

        self.conv13 = nn.Conv2d(256, 18, 1, 1, 1)
        self.bn13 = nn.BatchNorm2d(18, 0.9, 1e-5)

        #yolo2
        mask2 = [1,2,3]
        anchors2 = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
        classes2=1
        num2=6
        jitter2=.3
        ignore_thresh2 = .7
        truth_thresh2 = 1
        random2=1
        self.yolo2 = yolo(mask2, anchors2, classes2, num2, jitter2, ignore_thresh2, truth_thresh2, random2, self.height)


    def forward(self, input):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.mp2(x)

        route2 = x
        #x = self.block1.forward(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.mp3(x)

        # for i in range(0,2):
        #     x = self.block2.forward(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = self.mp4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = self.mp5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.activation(x)
        x = self.mp6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.activation(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.activation(x)

        x = self.conv9(x)
        x = self.bn9(x)
        route1 = x
        x = self.activation(x)

        x = self.conv10(x)
        x = self.bn10(x)

        #ask deep about the wheter -4 route will count yolo layers ar not
        x = self.yolo1(x)

        x = x + route1

        x = self.conv11(x)
        x = self.bn11(x)
        x = self.activation(x)

        x = self.upsample1(x)

        x = x+ route2

        x = self.conv12(x)
        x = self.bn12(x)
        x = self.activation(x)

        x = self.conv13(x)
        x = self.bn13(x)

        x = self.yolo2(x)
        

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()
