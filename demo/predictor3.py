# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import matplotlib.pyplot as plt

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from video_visualizer3 import VideoVisualizer
from visualizer3 import ColorMode, Visualizer

from adet.utils.visualizer import TextVisualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cfg = cfg
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.ROI_HEADS.NAME == "TextHead"

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
            print("cuda device ON")
        else:
            self.predictor = DefaultPredictor(cfg)
            print("CPU device ON")

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        if self.vis_text:
            visualizer = TextVisualizer(image, self.metadata, instance_mode=self.instance_mode, cfg=self.cfg)
        else:
            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "bases" in predictions:
            self.vis_bases(predictions["bases"])
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def vis_bases(self, bases):
        basis_colors = [[2, 200, 255], [107, 220, 255], [30, 200, 255], [60, 220, 255]]
        bases = bases[0].squeeze()
        bases = (bases / 8).tanh().cpu().numpy()
        num_bases = len(bases)
        fig, axes = plt.subplots(nrows=num_bases // 2, ncols=2)
        for i, basis in enumerate(bases):
            basis = (basis + 1) / 2
            basis = basis / basis.max()
            basis_viz = np.zeros((basis.shape[0], basis.shape[1], 3), dtype=np.uint8)
            basis_viz[:, :, 0] = basis_colors[i][0]
            basis_viz[:, :, 1] = basis_colors[i][1]
            basis_viz[:, :, 2] = np.uint8(basis * 255)
            basis_viz = cv2.cvtColor(basis_viz, cv2.COLOR_HSV2RGB)
            axes[i // 2][i % 2].imshow(basis_viz)
        plt.show()

    def run_on_video(self, video, img, classint):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions, img, classint):
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "instances" in predictions:
                predictions = predictions["instances"][predictions['instances'].pred_classes == classint].to(self.cpu_device)
                #vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
                test = predictions.pred_masks.numpy() #predictions의 마스크 영역만 가지고와서 numpy형태로 변환
                print("test.shape",test.shape)
                #trans = test.transpose((1,2,0))
                #print("trans.shape",trans.shape)
                print("frame 크기:",frame.shape,"img 크기:",img.shape,"\n")
                channel = test.shape[0]   #인스턴스 개수
                camwidth = test.shape[2]  #입력프레임의 width값
                camheight = test.shape[1] #입력프레임의 height값

                #background이미지와 입력프레임 영상 크기 비교하여 각각 보간
                if (frame.shape[0] * frame.shape[1]) == (img.shape[0] * img.shape[1]):
                    background = img
                    pass
                elif (frame.shape[0] * frame.shape[1]) > (img.shape[0] * img.shape[1]):
                    background = cv2.resize(img,dsize=(camwidth,camheight), interpolation=cv2.INTER_LINEAR)
                    print("background image upscaling",background.shape)
                else:
                    background = cv2.resize(img,dsize=(camwidth,camheight), interpolation=cv2.INTER_AREA)
                    print("background image downscaling",background.shape)
                              
                moutput = np.empty((camheight,camwidth)) #인스턴스가 2개이상일경우 각 인스턴스 mask영역 합치기 위한 임시 numpy
                #print(trans.shape)
               
                if channel == 0:          #predictions의 인스턴스 개수가 없으면 패스
                    print("no instance")
                    return background
                else:
                    i = 0
                    moutput = test[i]     #moutput에 입력프레임의 1번값 마스크로 초기화
                    for i in range(channel):
                        moutput = moutput + test[i]  #moutput에 각 인스턴스 값 or연산(내부 값들은 False,True로 되어있음)
                        #print("moutput shape",moutput.shape)  

                    moutput2 = moutput.reshape((camheight,camwidth,1)) #moutput의 형태는 2차원 배열이라 원본이미지와 대조하기위하여 3차원으로 변경
                    #print("moutput2 shape",moutput2.shape)    
                    #moutput3 = moutput.transpose((1,2,0))
                    #print("moutput3 shape",moutput3.shape)
                    #초록색 코드(57,254,20)
                    real1 = np.where(moutput2==True,frame,background) #입력프레임의 mask값이 True일경우 원본이미지 출력 아닐경우는 설정값에 따라변경
                    print(real1.shape)
                    real2 = np.array(real1, dtype=np.uint8) #이미지를 0~255형태의 인트형태로 변환
                    #trans = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
                    #cv2.imshow("real.jpg", real2)
                    #cv2.waitKey()
                    #cv2.destroyAllWindows()
                    
                """else:
                    print("Over One Instance")
                    i = 0
                    moutput = test[0,: : ]
                    for i in range(channel):
                        moutput = moutput + test[i,: : ]
                        print(moutput.shape)"""
                
                #print(moutput.shape)
            #vis_frame = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)   
            return real2

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions,img,classint)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions,img,classint)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame),img,classint)


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
