import os, sys
import ctypes
import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from pathlib import Path
pydir = Path(os.path.abspath(__file__)).parent
sys.path.append(f'{pydir.parent}/functions')
from general import xywh2xyxy, non_max_suppression

CONF_THRESH = 0.1
IOU_THRESHOLD = 0.4

class trt_wrapper(object):
    '''
    A class wraps tensorrt ops, inspired by wang-xinyu
    '''

    def __init__(self, engine_file_path, plugins_path, ori_imgsize):
        self.ori_imgsize = ori_imgsize
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        ctypes.CDLL(plugins_path)
        # 反序列化
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            # Engine contains 2 bindings: 'data' and 'prob' whose sizes are (3, 384, 640)
            # and (6001, 1, 1) respectively. engine.max_batch_size is declared when engine
            # created. trt volume is the opreation that mutiples each dims of bindings.
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs  # size: 6001
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, source, post_proc=False):
        source = source.numpy() if isinstance(source, torch.Tensor) else source
        assert isinstance(
            source, np.ndarray), f'Invalid image type. ndarray is expected, but got {type(source)}'
        self.cfx.push()
        context = self.context
        bindings = self.bindings
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        stream = self.stream
        if source.ndim == 3:
            # (c, h, w)
            np.copyto(host_inputs[0], source.ravel())
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            context.execute_async(
            	bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
            stream.synchronize()
            self.cfx.pop()
            output = host_outputs[0]
            if post_proc is False:
                return outputs
            else:
                result_boxes, result_scores, result_classid = self.post_process(
                    output)
                pred = torch.cat(
                    (result_boxes, result_scores[:, np.newaxis], result_classid[:, np.newaxis]), axis=1)
                return [pred]
        if source.ndim == 4:  # batched inference (batch, c, w, h)
            batch_size = source.shape[0]
            size = source[0].ravel().shape[0]
            flag = 0
            for img in source:
                np.copyto(host_inputs[0][flag*size:(flag+1)
                          * size], source[flag].ravel())
                flag += 1
            host_inputs[0] = host_inputs[0].reshape(
            	batch_size, source[0].ravel().shape[0])
            host_outputs[0] = host_outputs[0].reshape(batch_size, 6001)
            # transfer input data into GPU
            [cuda.memcpy_htod_async(cuda_inputs[0], host_input, stream)
             for host_input in host_inputs[0]]
            context.execute_async(
            	bindings=bindings, stream_handle=stream.handle)
            [cuda.memcpy_dtoh_async(host_output, cuda_outputs[0], stream)
             for host_output in host_outputs[0]]
            stream.synchronize()
            self.cfx.pop()
            outputs = host_outputs[0]  # array (batch, 6001)
            if post_proc is False:
                return outputs
            else:
                pred = []
                for output in outputs:  # for single inference, outputs length is 1
                    result_boxes, result_scores, result_classid = self.post_process(
                    	output)
                    pred.append(torch.cat(
                        result_boxes, result_scores[:, np.newaxis], result_classid[:, np.newaxis]))
                return pred

    def post_process(self, output):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # reshape to a 2 dimension array
        pred = np.reshape(output[1:], (-1, 6))[:num, :]  # [526, 6]
        # to a torch Tensor
        pred = torch.Tensor(pred).cuda()
        # Get the boxes
        # boxes position; torch.Size([526,4]); get the first 4 cols
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]  # confidence; torch.Size([526]); get the 5th col
        # Get the classid
        classid = pred[:, 5]  # classid
        # Choose those boxes that score > CONF_THRESH
        si = scores > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Transform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        # boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        boxes = xywh2xyxy(boxes, need_scale=True,
                          im0=np.zeros(self.ori_imgsize))
        # Do nms
        # indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        indices = non_max_suppression(boxes, scores, IOU_THRESHOLD)
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid
