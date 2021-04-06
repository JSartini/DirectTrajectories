from __future__ import division

import gc
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import math
#from torchsummary import summary

# Generic Functions
def parse_cfg(cfg_file):
    file = open(cfg_file, "r")
    network_blocks = []
    curr_block = {}
    for line in file:
        line = line.rstrip().lstrip()
        if(len(line) > 0): # Remove empty lines
            if(line[0] != "#"): # Comment and empty lines removed
                if(line[0]=="["): # New module
                    if(len(curr_block) > 0): # Take care of previous block dict
                        network_blocks.append(curr_block)
                        curr_block = {}
                    curr_block["type"] = line[1:-1]
                else:
                    key,val = line.split("=")
                    curr_block[key] = val
    network_blocks.append(curr_block) # Save the final block
    return network_blocks

class route_layer(nn.Module):
    def __init__(self, where):
        super(route_layer, self).__init__()
        self.where = where

class reorg_layer(nn.Module):
    def __init__(self, hcut, wcut):
        super(reorg_layer, self).__init__()
        self.hcut = hcut
        self.wcut = wcut

    # Adapted from
    # https://github.com/ayooshkathuria/PyTorch-YOLO-v2/blob/master/darknet.py
    def forward(self,x):
        B,C,H,W = x.data.shape
        newH = H//self.hcut
        newW = W//self.wcut
        x = x.view(B,C, newH, self.hcut, newW, self.wcut).transpose(-2,-3).contiguous()
        x = x.view(B,C, newH * newW, self.hcut, self.wcut)
        x = x.view(B,C, newH * newW, self.hcut*self.wcut).transpose(-1,-2).contiguous()
        x = x.view(B, C, self.wcut*self.hcut, newH, newW).transpose(1,2).contiguous()
        x = x.view(B, C*self.wcut*self.hcut, newH, newW)
        return x

def create_network(spec, init_features):
    mod_list = nn.ModuleList()
    feature_layers = init_features
    feat_layer_list = []

    layer = 0
    for block in spec:
        module = nn.Sequential()

        if(block["type"] == "convolutional"):
            if("batch_normalize" in block.keys()):
                norm = int(block["batch_normalize"])
                bias = False
            else:
                norm = 0
                bias = True

            num_features = int(block["filters"])
            k_size = int(block["size"])
            stride = int(block["stride"])
            pad = int(block["pad"])
            if(pad):
                pad = (k_size-1)//2 # Same padding
            else:
                pad = 0

            # Perform convolution
            conv = nn.Conv2d(feature_layers, num_features, k_size, stride, pad, bias=bias)
            module.add_module("Convolution", conv)
            # Perform normalization if indicated
            if(norm>0):
                normal = nn.BatchNorm2d(num_features)
                module.add_module("Batch Normalization", normal)

            activation = block["activation"]
            if(activation=="leaky"):
                active = nn.LeakyReLU(inplace=True)
                module.add_module("LReLU", active)
            elif(activation=="relu"):
                active = nn.ReLU(inplace=True)
                module.add_module("ReLU", active)
            feature_layers = num_features

        elif(block["type"] == "lstm"):
            insize = int(block["input_size"])
            hsize = int(block["hidden_size"])
            nlayers = int(block["num_layers"])
            lstm = LSTM_mod(insize, hsize, nlayers)
            module.add_module("LSTM", lstm)
            linear = nn.Linear(hsize, insize)
            module.add_module("Linear",linear)
            feature_layers = feat_layer_list[-1]

        elif(block["type"] == "maxpool"):
            stride = int(block["stride"])
            size = int(block["size"])
            mpool = nn.MaxPool2d(size, stride= stride)
            module.add_module("Max Pool", mpool)

            feature_layers = feat_layer_list[-1]

        elif(block["type"] == "route"):
            layers = block["path"].split(",")
            layers = [int(num) for num in layers]
            route = route_layer(layers)
            module.add_module("Route", route)
            feature_layers = feat_layer_list[layers[0]]
            if(len(layers) > 1):
                feature_layers += feat_layer_list[layers[1]]

        else: # Reorg case
            stride = int(block["stride"])
            hcut = int(block["height_cut"])
            wcut = int(block["width_cut"])
            reorg = reorg_layer(hcut, wcut)
            module.add_module("Reorganization", reorg)

            feature_layers = feat_layer_list[-1]*hcut*wcut

        feat_layer_list.append(feature_layers)
        mod_list.append(module)
        layer += 1
    return mod_list

class Model(nn.Module):
  def __init__(self, mod_list):
    super(Model, self).__init__()
    self.mod_list = mod_list

  def forward(self, x):
    outputs = []
    for mod in self.mod_list:
        g = mod.named_modules()
        next(g) # Burn first blank space
        name = next(g)[0]
        if(name != "Route" and name != "LSTM"):
            x = mod(x)
            outputs.append(x)

        elif(name == "LSTM"):
            B, C, H, W = x.shape
            x = x.reshape((B,C*H*W)).unsqueeze(1)
            x = mod.LSTM(x)
            x = mod.Linear(x.squeeze())
            x = x.reshape((B,C,H,W))
            outputs.append(x)

        else:
            m = mod.children()
            info_mod = next(m)
            locations = info_mod.where
            if(len(locations) == 1):
                x = outputs[locations[0]]
            else:
                first, second = locations
                x = torch.cat((outputs[first], outputs[second]), 1)
            outputs.append(x)
    return x

# YOLO-specific functions (some inspiration from https://github.com/kuangliu/pytorch-yolov2/blob/master/loss.py)
def calc_centers(image_size, batch_size):
    H,W = image_size
    x = torch.arange(W).unsqueeze(0)
    y = torch.arange(H).unsqueeze(1)
    xs = x.repeat(H,1).unsqueeze(0)
    ys = y.repeat(1,W).unsqueeze(0)
    idx = torch.cat((xs, ys), dim=0).unsqueeze(0)
    out = idx.repeat(batch_size,1,1,1)
    return out

def anchor_boxes(anchors, image_size, batch_size):
    num_anchors = len(anchors)
    out = torch.zeros((batch_size, num_anchors, 2, *image_size))
    for i in range(num_anchors):
        out[:,i,0,:,:] = anchors[i,0]
        out[:,i,1,:,:] = anchors[i,1]
    return out

def index_forward(anchor, row, col, image_size):
    H = image_size[0]
    W = image_size[1]
    return anchor*(H*W) + row*(W) + col

def index_reverse(ind, H, W):
    anchor = ind // (H*W)
    new_ind = ind - anchor*(H*W)
    row = new_ind // W
    col = new_ind - row*W
    return (anchor, row, col)

def create_boxlist(tensor, num_anchors, image_size):
    H = image_size[0]
    W = image_size[1]
    num_boxes = num_anchors*H*W
    boxes = torch.zeros((num_boxes,4))
    scores = torch.zeros(num_boxes)
    for i in range(num_anchors):
        for j in range(H):
            for k in range(W):
                if(tensor[i,2,j,k] != 0 and tensor[i,3,j,k] != 0):
                    index = index_forward(i, j, k, image_size)
                    boxes[index,:] = tensor[i,:4,j,k]
                    scores[index] = tensor[i,4,j,k]
    out_boxes = torch.zeros((num_boxes,4))
    out_boxes[:,0] = boxes[:,0]-boxes[:,2]/2
    out_boxes[:,1] = boxes[:,1]-boxes[:,3]/2
    out_boxes[:,2] = boxes[:,0]+boxes[:,2]/2
    out_boxes[:,3] = boxes[:,1]+boxes[:,3]/2
    return out_boxes, scores

def transform_output_view(net_out, obj_thresh, nms_thresh, anchors, res):
    # Get requisite data and reshape
    B, C, H, W = net_out.shape
    num_anchors = anchors.shape[0]

    out = net_out.view(B, num_anchors, C//num_anchors, H, W)
    num_classes = C//num_anchors - 5

    oloc = torch.zeros((B,num_anchors,4,H,W))
    oclass = torch.zeros((B,num_anchors,num_classes,H,W))
    oconf = torch.zeros((B,num_anchors,1,H,W))

    # First transform outputs to ind. boxes and scale all parameters appropriately
    centers = calc_centers((H,W), B)*res
    out[:,:,:2,:,:] = (torch.sigmoid(out[:,:,:2,:,:]) +
                        centers.unsqueeze(1).expand_as(out[:,:,:2,:,:]))
    anch_dims = anchor_boxes(anchors, (H,W), B)
    out[:,:,2:4,:,:] = torch.exp(out[:,:,2:4,:,:]) * anch_dims
    out[:,:,4,:,:] = torch.sigmoid(out[:,:,4,:,:])

    # Perform on an image-by-image basis
    for i in range(B):
        image_pred = out[i,:,:,:,:]
        # Perform thresholding on object score for all points/anchors
        ot_mask = (image_pred[:,4,:,:] > obj_thresh).float().unsqueeze(1)
        image_pred = image_pred*ot_mask

        # Perform non-max suppression class-wise, have to first extract class/score
        scores, classes = torch.max(image_pred[:,5:,:,:], 1)
        scores = scores.unsqueeze(1)
        classes = classes.unsqueeze(1)
        image_pred = torch.cat((image_pred[:,:,:,:], classes, scores),1)
        tensors = []
        for j in range(num_classes):
            tensor = image_pred*(image_pred[:,9,:,:] == num_classes).float().unsqueeze(1)
            boxes, scores = create_boxlist(tensor, num_anchors, (H,W))
            indices = torchvision.ops.nms(boxes, scores, nms_thresh)
            for k in range(len(boxes)):
                if(k not in indices):
                    anch, row, col = index_reverse(indices[k], H, W)
                    tensor[anch,:,row,col] = 0
            tensors.append(tensor.unsqueeze(0))
        tensors = torch.cat(tensors, 0)
        tensors = torch.sum(tensors, 0)

        oloc[i,:,:,:,:] = tensors[:,:4,:,:]
        oclass[i,:,:,:,:] = tensors[:,5:9,:,:]
        oconf[i,:,:,:,:] = tensors[:,4,:,:].unsqueeze(1)
    return oloc, oclass, oconf

# Intersection over Union functions for use in Bounding Box operations
def size_iou(dim1, dim2):
  intersect = np.prod(np.minimum(dim1, dim2))
  return intersect/(np.prod(dim1) + np.prod(dim2) - intersect)

def box_iou(box1, box2):
    num_boxes_1 = box1.shape[0]
    num_boxes_2 = box2.shape[0]
    box1tl = torch.cat(((box1[:,0]-box1[:,2]/2).unsqueeze(1),
                (box1[:,1]-box1[:,3]/2).unsqueeze(1)), -1)
    box2tl = torch.cat(((box2[:,0]-box2[:,2]/2).unsqueeze(1),
                (box2[:,1]-box2[:,3]/2).unsqueeze(1)), -1)
    box1br = torch.cat(((box1[:,0]+box1[:,2]/2).unsqueeze(1),
                (box1[:,1]+box1[:,3]/2).unsqueeze(1)), -1)
    box2br = torch.cat(((box2[:,0]+box2[:,2]/2).unsqueeze(1),
                (box2[:,1]+box2[:,3]/2).unsqueeze(1)), -1)
    box1tl = box1tl.unsqueeze(1).expand(num_boxes_1, num_boxes_2, 2)
    box2tl = box2tl.unsqueeze(0).expand(num_boxes_1, num_boxes_2, 2)
    box1br = box1br.unsqueeze(1).expand(num_boxes_1, num_boxes_2, 2)
    box2br = box2br.unsqueeze(0).expand(num_boxes_1, num_boxes_2, 2)
    tl = torch.maximum(box1tl, box2tl)
    br = torch.minimum(box1br, box2br)
    sides = torch.clamp(br-tl, min=0)
    intersect = sides[:,:,0]*sides[:,:,1]
    b1sides = torch.clamp(box1br-box1tl, min=0)
    b1areas = b1sides[:,:,0]*b1sides[:,:,1]
    b2sides = torch.clamp(box2br-box2tl, min=0)
    b2areas = b2sides[:,:,0]*b2sides[:,:,1]
    return intersect/(b1areas + b2areas - intersect)

def elem_iou(box1, box2):
    box1tl = torch.cat(((box1[:,:,0,:,:]-box1[:,:,2,:,:]/2).unsqueeze(2),
                (box1[:,:,1,:,:]-box1[:,:,3,:,:]/2).unsqueeze(2)), 2)
    box2tl = torch.cat(((box2[:,:,0,:,:]-box2[:,:,2,:,:]/2).unsqueeze(2),
                (box2[:,:,1,:,:]-box2[:,:,3,:,:]/2).unsqueeze(2)), 2)
    box1br = torch.cat(((box1[:,:,0,:,:]+box1[:,:,2,:,:]/2).unsqueeze(2),
                (box1[:,:,1,:,:]+box1[:,:,3,:,:]/2).unsqueeze(2)), 2)
    box2br = torch.cat(((box2[:,:,0,:,:]+box2[:,:,2,:,:]/2).unsqueeze(2),
                (box2[:,:,1,:,:]+box2[:,:,3,:,:]/2).unsqueeze(2)), 2)
    tl = torch.maximum(box1tl, box2tl)
    br = torch.minimum(box1br, box2br)
    sides = torch.clamp(br-tl, min=0)
    intersect = sides[:,:,0,:,:]*sides[:,:,1,:,:]
    b1sides = torch.clamp(box1br-box1tl, min=0.000001)
    b1areas = b1sides[:,:,0,:,:]*b1sides[:,:,1,:,:]
    b2sides = torch.clamp(box2br-box2tl, min=0.000001)
    b2areas = b2sides[:,:,0,:,:]*b2sides[:,:,1,:,:]
    return intersect/(b1areas + b2areas - intersect)

def transform_train(training_labels, res, anchors, H, W, cuda, B):
    num_anchors = anchors.shape[0]
    image_idxs = torch.unique(training_labels[:,-1])
    batch_size = image_idxs.shape[0]

    locs = torch.zeros((batch_size, num_anchors, 4, H, W))
    classes = torch.ones((batch_size, num_anchors, 1, H, W))*-1
    confidence = torch.zeros((batch_size, num_anchors, 1, H, W))

    if(cuda):
        locs = locs.cuda()
        classes = classes.cuda()
        confidence = confidence.cuda()

    for i in range(batch_size):
        this_img = training_labels[training_labels[:,-1]==image_idxs[i], :]
        # Case of no items
        if(this_img[0, -2].item() == -1):
            continue

        # Case of items
        for j in range(this_img.shape[0]):
            box_loc = this_img[j,:4].float()
            box_class = this_img[j,4].float()

            # First, find the correct cell
            x,y = box_loc[:2]
            x = x.item()
            y = y.item()
            xcell, ycell = int(x//res), int(y//res)

            # Determine max iou anchor box
            anch_xy = box_loc[:2]
            anchs = torch.cat((anch_xy-anchors/2, anch_xy+anchors/2), 1)
            ious = box_iou(box_loc.unsqueeze(0), anchs)
            _, ind = ious.max(1)

            # Scale down the box center locations
            box_loc = box_loc/res

            # Save appropriate cell information in th correct location
            locs[i,ind,:,ycell,xcell] = box_loc
            classes[i,ind,0,ycell,xcell] = box_class
            confidence[i,ind,0,ycell,xcell] = 1

    return locs, classes, confidence

class YOLO_loss(nn.Module):
    def __init__(self, lam_coord, lam_noobj, lam_obj, lam_class, resolution, anchors, image_size, cuda, B):
        self.lam_coord = lam_coord
        self.lam_noobj = lam_noobj
        self.lam_obj = lam_obj
        self.lam_class = lam_class
        self.resolution = resolution
        H,W = image_size
        if(cuda):
            self.anchors = anchors.cuda()
            self.centers = calc_centers((H,W),B).cuda()
            self.anch_dims = (anchor_boxes(anchors, (H,W), B)/resolution).cuda()
        else:
            self.anchors = anchors
            self.centers = calc_centers((H,W),B)
            self.anch_dims = anchor_boxes(anchors, (H,W), B)/resolution
        self.image_size = image_size
        self.cuda = cuda
        super(YOLO_loss, self).__init__()

    def forward(self, output, train_labels):
        B, C, H, W = output.shape
        true_locs, true_classes, true_conf = transform_train(train_labels,
                            self.resolution, self.anchors, *self.image_size, self.cuda, B)
        true_xy = true_locs[:,:,:2,:,:]
        true_wh = true_locs[:,:,2:4,:,:].sqrt()
        obj_mask = (true_conf > 0)
        noobj_mask = (true_conf == 0)

        # Perform transforms required to use network output
        num_anchors = self.anchors.shape[0]

        output = output.view(B, num_anchors, C//num_anchors, H, W).clone()
        num_classes = C//num_anchors - 5

        pred_xy = torch.sigmoid(output[:,:,:2,:,:])
        pred_xy = pred_xy + self.centers.unsqueeze(1).expand_as(pred_xy)
        pred_wh = (torch.exp(output[:,:,2:4,:,:]) * self.anch_dims).sqrt()
        pred_conf = torch.sigmoid(output[:,:,4,:,:]).unsqueeze(2)
        pred_classes = output[:,:,5:,:,:]

        # Calc IOU for true confidence values:
        iou = elem_iou(torch.cat((pred_xy*self.resolution, pred_wh*self.resolution),dim=2),
                        true_locs).unsqueeze(2)
        if(self.cuda):
            iou = iou.cuda()

        if(torch.sum(obj_mask) == 0):
            noobj_loss = F.mse_loss(pred_conf[noobj_mask], iou[noobj_mask])
            return self.lam_noobj*noobj_loss

        else:
            # Calc location loss
            xy_loss = F.mse_loss(pred_xy[obj_mask.expand_as(pred_xy)],
                            true_xy[obj_mask.expand_as(true_xy)])
            wh_loss = F.mse_loss(pred_wh[obj_mask.expand_as(pred_wh)],
                            true_wh[obj_mask.expand_as(true_wh)])

            # Calc confidence loss
            obj_loss = F.mse_loss(pred_conf[obj_mask], iou[obj_mask])
            noobj_loss = F.mse_loss(pred_conf[noobj_mask], iou[noobj_mask])

            # Calc class loss
            tc = true_classes[obj_mask].long()
            pc = pred_classes[obj_mask.expand_as(pred_classes)].view(tc.shape[0],5)
            class_loss = F.cross_entropy(pc, tc)

            return (self.lam_coord*(xy_loss+wh_loss) + self.lam_obj*obj_loss +
                    self.lam_noobj*noobj_loss + self.lam_class*class_loss)

# Novel Network-specific Functions
def novel_train(training_labels, image_size, anchors, res, cuda, B):
    num_anchors = anchors.shape[0]
    H, W = image_size
    image_idxs = torch.unique(training_labels[:,-1])
    batch_size = image_idxs.shape[0]

    locs = Variable(torch.zeros((batch_size, num_anchors, 4, H, W)), requires_grad=False)
    trajectory = Variable(torch.zeros((batch_size, num_anchors, 4, H, W)), requires_grad=False)
    obj_present = Variable(torch.zeros((batch_size, num_anchors, 1, H, W)), requires_grad=False)

    if(cuda):
        locs = locs.cuda()
        trajectory = trajectory.cuda()
        obj_present = obj_present.cuda()

    for i in range(batch_size):
        this_img = training_labels[training_labels[:,-1]==image_idxs[i], :]
        # Case of no items
        if(this_img[0, -2].item() == -1):
            continue

        # Case of items
        for j in range(this_img.shape[0]):
            box_loc = this_img[j,:4].float()
            box_traj = this_img[j,4:8].float()

            # First, find the correct cell
            x,y = box_loc[:2]
            x = x.item()
            y = y.item()
            xcell, ycell = int(x//res), int(y//res)

            # Determine max iou anchor box
            anch_xy = box_loc[:2]
            anchs = torch.cat((anch_xy-anchors/2, anch_xy+anchors/2), 1)
            ious = box_iou(box_loc.unsqueeze(0), anchs)
            _, ind = ious.max(1)

            # Calculate real object's location and box size according to YOLO
            box_loc = box_loc/res

            # Save appropriate cell information in the correct location
            locs[i,ind,:,ycell,xcell] = box_loc
            trajectory[i,ind,:,ycell,xcell] = box_traj
            obj_present[i,ind,0,ycell,xcell] = 1

    return locs, trajectory, obj_present

class Novel_loss(nn.Module):
    def __init__(self, anchors, lam_bl, lam_ov, lam_oa, lam_obj, lam_noobj, resolution, image_size, cuda, B):
        self.lam_bl = lam_bl
        self.lam_ov = lam_ov
        self.lam_oa = lam_oa
        self.lam_obj = lam_obj
        self.lam_noobj = lam_noobj
        self.resolution = resolution
        self.image_size = image_size
        H,W = image_size
        if(cuda):
            self.anchors = anchors.cuda()
            self.centers = calc_centers((H,W),B).cuda()
            self.anch_dims = (anchor_boxes(anchors, (H,W), B)/resolution).cuda()
        else:
            self.anchors = anchors
            self.centers = calc_centers((H,W),B)
            self.anch_dims = anchor_boxes(anchors, (H,W), B)/resolution
        self.cuda = cuda
        super(Novel_loss, self).__init__()

    def forward(self, out, train_labels):
        B, C, H, W = out.shape
        true_locs, true_traj, o_m = novel_train(train_labels, self.image_size,
                                self.anchors,self.resolution, self.cuda, B)

        obj_mask = (o_m == 1)
        noobj_mask = (o_m == 0)

        # Perform transforms required to use network output
        num_anchors = self.anchors.shape[0]
        output = out.view(B, num_anchors, C//num_anchors, H, W).clone()
        output[:,:,:2,:,:] = torch.sigmoid(output[:,:,:2,:,:])
        output[:,:,4,:,:] = torch.sigmoid(output[:,:,4,:,:])

        # Calc object location and trajectory loss
        box_loc_true = true_locs[:,:,:2,:,:]
        box_size_true = true_locs[:,:,2:,:,:].sqrt()
        obj_vel_true = true_traj[:,:,:2,:,:]
        obj_acc_true = true_traj[:,:,2:,:,:]

        box_loc_pred = torch.sigmoid(output[:,:,:2,:,:])
        box_loc_pred = box_loc_pred + self.centers.unsqueeze(1).expand_as(box_loc_pred)
        box_size_pred = (torch.exp(output[:,:,2:4,:,:]) * self.anch_dims).sqrt()
        pred_obj = torch.sigmoid(output[:,:,4,:,:]).unsqueeze(2)
        obj_vel_pred = output[:,:,5:7,:,:]
        obj_acc_pred = output[:,:,7:,:,:]

        iou = elem_iou(torch.cat((box_loc_pred*self.resolution, box_size_pred*self.resolution),dim=2),
                true_locs[:,:,:4,:,:]).unsqueeze(2)
        if(self.cuda):
            iou = iou.cuda()

        if(torch.sum(obj_mask) == 0):
            noobj_loss = F.mse_loss(iou[noobj_mask], pred_obj[noobj_mask])
            return self.lam_noobj*noobj_loss

        else:
            box_loc_loss = F.mse_loss(box_loc_true[obj_mask.expand_as(box_loc_true)],
                                    box_loc_pred[obj_mask.expand_as(box_loc_pred)])
            box_size_loss = F.mse_loss(box_size_true[obj_mask.expand_as(box_size_true)],
                                    box_size_pred[obj_mask.expand_as(box_size_pred)])
            obj_vel_loss = F.mse_loss(obj_vel_true[obj_mask.expand_as(obj_vel_true)],
                                    obj_vel_pred[obj_mask.expand_as(obj_vel_pred)])
            obj_acc_loss = F.mse_loss(obj_acc_true[obj_mask.expand_as(obj_acc_true)],
                                    obj_acc_pred[obj_mask.expand_as(obj_acc_pred)])
            obj_loss = F.mse_loss(iou[obj_mask], pred_obj[obj_mask])
            noobj_loss = F.mse_loss(iou[noobj_mask], pred_obj[noobj_mask])

            return (self.lam_bl * (box_loc_loss + box_size_loss) + self.lam_ov * obj_vel_loss +
                self.lam_oa * obj_acc_loss + self.lam_obj * obj_loss + self.lam_noobj * noobj_loss)

# LSTM-Network specific functions
def shape_for_LSTM(yolo_output, cuda):
    B, C, H, W = yolo_output.shape
    sequence_length = C*H*W
    output = Variable(torch.zeros((B-1, 2, sequence_length)))
    for i in range(B-1):
        output[i,0,:] = yolo_output[i,:,:,:].flatten()
        output[i,1,:] = yolo_output[i+1,:,:,:].flatten()
    if(cuda):
        output = output.cuda()
    return output

def transform_LSTM_train(training_labels, res, image_size, cuda):
    H, W = image_size
    image_idxs = torch.unique(training_labels[:,-1])
    batch_size = image_idxs.shape[0]

    trajectory = Variable(torch.zeros((batch_size-1, 4, H, W)), requires_grad=False)
    obj_present = Variable(torch.zeros((batch_size-1, 1, H, W)), requires_grad=False)

    if(cuda):
        trajectory = trajectory.cuda()
        obj_present = obj_present.cuda()

    for i in range(1,batch_size):
        this_img = training_labels[training_labels[:,-1]==image_idxs[i], :]

        # Case of no items
        if(this_img[0, -2].item() == -1):
            continue

        # Case of items
        for j in range(this_img.shape[0]):
            box_loc = this_img[j,:4].float()
            box_traj = this_img[j,4:8].float()

            # First, find the correct cell
            x,y = box_loc[:2]
            x = x.item()
            y = y.item()
            xcell, ycell = int(x//res), int(y//res)

            # Save appropriate cell information in th correct location
            trajectory[i-1,:,ycell,xcell] = box_traj
            obj_present[i-1,0,ycell,xcell] = 1

    return trajectory, obj_present

class LSTM_module(nn.Module):
    def __init__(self, input_dim, hidden_dim, nlayer, target_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim,
                        num_layers = nlayer, batch_first = True)
        self.to_out = nn.Linear(hidden_dim, target_dim)

    def forward(self,x):
        lstm_out, _ = self.lstm(x)
        output = self.to_out(lstm_out[:,1,:].squeeze())
        return output

class LSTM_loss(nn.Module):
    def __init__(self, lam_noobj, lam_accel, lam_vel, resolution, image_size, cuda):
        self.lam_noobj = lam_noobj
        self.lam_accel = lam_accel
        self.lam_vel = lam_vel
        self.resolution = resolution
        self.image_size = image_size
        self.cuda = cuda
        super(LSTM_loss, self).__init__()

    def forward(self, output, train_labels):
        true_traj, o_m = transform_LSTM_train(train_labels,
                                                self.resolution, self.image_size, self.cuda)

        train_vel = true_traj[:,:2,:,:]
        train_accel = true_traj[:,2:,:,:]
        obj_mask = (o_m > 0)
        noobj_mask = (o_m == 0)

        pred_vel = output[:,:2,:,:]
        pred_accel = output[:,2:4,:,:]
        pred_obj = output[:,-1,:,:].unsqueeze(1)

        if(torch.sum(obj_mask) == 0):
            noobj_loss = F.mse_loss(pred_obj[noobj_mask], o_m[noobj_mask])
            return self.lam_noobj* noobj_loss

        else:
            # Calc confidence loss
            noobj_loss = F.mse_loss(pred_obj[noobj_mask], o_m[noobj_mask])

            # Calc vel and accel losses
            vel_loss = F.mse_loss(pred_vel[obj_mask.expand_as(pred_vel)],
                    train_vel[obj_mask.expand_as(train_vel)])
            accel_loss = F.mse_loss(pred_accel[obj_mask.expand_as(pred_accel)],
                    train_accel[obj_mask.expand_as(train_accel)])

            return (self.lam_noobj*noobj_loss+self.lam_vel*vel_loss+self.lam_accel*accel_loss)

def centerwh_to_xy(box):
    out_box = np.zeros(4)
    x1 = (box[0,:]-box[2,:]/2).unsqueeze(1)
    x2 = (box[0,:]+box[2,:]/2).unsqueeze(1)
    y1 = (box[1,:]-box[3,:]/2).unsqueeze(1)
    y2 = (box[1,:]+box[3,:]/2).unsqueeze(1)
    out = torch.cat((x1,x2,y1,y2), 1)
    return out

def nms_thresh(yolo_output, threshold, anchors, cuda, res):
    B, C, H, W = yolo_output.shape
    num_anchors = anchors.shape[0]
    centers = calc_centers((H,W),B)
    anch_dims = anchor_boxes(anchors, (H,W), B)
    if(cuda):
        centers = centers.cuda()
        anch_dims = anch_dims.cuda()

    # Perform post-processing transforms
    out = yolo_output.view(B, num_anchors, C//num_anchors, H, W).detach().clone()
    out[:,:,:2,:,:] = torch.sigmoid(out[:,:,:2,:,:])
    out[:,:,:2,:,:] = (out[:,:,:2,:,:] + centers.unsqueeze(1).expand_as(out[:,:,:2,:,:]))*res
    out[:,:,2:4,:,:] = torch.exp(out[:,:,2:4,:,:]) * anch_dims
    out[:,:,4,:,:] = torch.sigmoid(out[:,:,4,:,:])

    # Run the NMS
    t_output = out.transpose(1,2)
    for i in range(B):
        mask = torch.zeros(num_anchors*H*W).contiguous()
        this_entry = t_output[i,...].squeeze().detach().clone()
        boxes = t_output[i,:4,:,:,:].squeeze().contiguous()
        boxes = boxes.view(4,-1)
        boxes = centerwh_to_xy(boxes)
        scores = t_output[i,4,:,:,:].contiguous()
        scores = scores.view(1,-1).squeeze()
        idxs = torchvision.ops.nms(boxes, scores, 0.2)
        mask[idxs] = 1
        mask[scores < threshold] = 0
        mask = mask.view(num_anchors, H, W).unsqueeze(0).expand_as(this_entry)
        this_entry[(mask == 0)] = 0
        out[i,...] = this_entry.transpose(0,1)

    centers = None
    anch_dims = None
    gc.collect()
    if(cuda):
      torch.cuda.empty_cache()
    return out.view(B,C,H,W)

# Trajectory Calculations from YOLO output
def req_info(yolo_output, anchors, cuda):
    B, C, H, W = yolo_output.shape
    num_anchors = anchors.shape[0]
    output = yolo_output.view(B, num_anchors, C//num_anchors, H, W).detach().clone()
    no_obj_mask = (output[:,:,1,:,:].detach()==0).unsqueeze(2)

    boxes = output[:,:,:4,:,:]
    boxes[no_obj_mask.expand_as(boxes)] = 0
    func = torch.nn.Softmax(dim=2)
    pred_classes = func(output[:,:,5:,:,:])
    _, pred_class = torch.max(pred_classes, 2)
    pred_conf = output[:,:,4,:,:]

    return boxes, pred_class, pred_conf

def nearest(prev_box, next_boxes, threshold):
    distances = torch.cdist(prev_box, next_boxes)[0]
    if(distances.shape[0] == 0):
        return -1
    min_dist, cont_point = torch.min(distances, dim=0)
    if(min_dist < threshold):
        return cont_point.item()
    else:
        return -1

def match_boxes(yolo_seq, anchors, threshold, image_size, cuda):
    B, C, H, W = yolo_seq.shape
    A = anchors.shape[0]
    output = torch.zeros(((B-1),A*H*W,2))
    all_boxes, all_classes, _ = req_info(yolo_seq, anchors, cuda)

    init_boxes = all_boxes[0,...].transpose(0,1).contiguous()
    init_boxes = init_boxes.view(4,-1).transpose(0,1)
    init_objs = torch.where(init_boxes[:,1]!=0)[0]
    init_boxes = init_boxes[init_objs,:]
    init_classes = all_classes[0,...].contiguous().view(1,-1).squeeze()
    init_classes = init_classes[init_objs]

    for i in range(1,B):
        second_boxes = all_boxes[i,...].transpose(0,1).contiguous()
        second_boxes = second_boxes.view(4,-1).transpose(0,1)
        second_objs = torch.where(second_boxes[:,1]!=0)[0]
        second_boxes = second_boxes[second_objs, :]
        second_classes = all_classes[i,...].contiguous().view(1,-1).squeeze()
        second_classes = second_classes[second_objs]

        for j in range(init_boxes.shape[0]):
            class_align = torch.where(second_classes == init_classes[j])[0]
            potential = second_boxes[class_align, :2]
            des_idx = nearest(init_boxes[j,:2].unsqueeze(0), potential[:,:2], threshold)
            if(des_idx != -1):
                over_arch_idx = second_objs[class_align[des_idx]]
                #subset_2xy = second_xy[:,class_align]
                #output[(i-1),over_arch_idx,:] = (subset_2xy[:,des_idx] - init_xy[:,j])*10
                output[(i-1),over_arch_idx,:] = (potential[des_idx,:] - init_boxes[j,:2])#*10

        init_boxes = second_boxes
        init_classes = second_classes
        #init_xy = second_xy

    return output.view((B-1),A,W,H,2).transpose(2,4)
