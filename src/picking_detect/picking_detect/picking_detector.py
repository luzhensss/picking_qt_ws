#!/usr/bin/python3

import json
import rclpy
from rclpy.node import Node 
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import os
import sys
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation as R

from .enhance import enhance_frame_bgr_greedy

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "请先安装 onnxruntime:\n"
        "  pip install onnxruntime          (CPU)\n"
        "  pip install onnxruntime-gpu      (GPU)"
    )

# ===================== 模型配置 (来自 picking_detector_onnx.py) =====================

# ONNX 模型路径和输入输出配置
ONNX_MODEL_PATH = '/home/luzhens/lzs/nanodet_models/fruit_0228/fruit/nanodet_fruit.onnx'
INPUT_HEIGHT = 640
INPUT_WIDTH = 480

# 性能调优
# 相机分辨率: 与模型输入一致可减少 resize；需要高分辨率可改 1280,720
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# 显示缩放 0.5=缩小一半降低开销，1.0=原尺寸
DISPLAY_SCALE = 1
# 不显示窗口可提升约 2–3 FPS（无头模式）
USE_DISPLAY = True

# 低光照贪心分块增强（见 picking_detect/enhance.py；与选题「关键区域优先」一致：更暗块优先增强）
USE_LOW_LIGHT_ENHANCE = True
LOW_LIGHT_BLOCK_SIZE = 32
LOW_LIGHT_BRIGHTNESS_THRESH = 80.0
LOW_LIGHT_MAX_GAIN = 2.5

# ===================== 业务配置 =====================
# 视觉模式:
# - "onnx": 使用当前目标检测逻辑
# - "qr":   预留给二维码识别模式，当前会跳过ONNX推理
VISION_MODE = "onnx"

# 作业区域:
# - "A": 抓取成熟果，类别名前缀为 c
# - "C": 只抓和目标标识一致的类别
WORK_REGION = "C"

# 调试模式下保留“按空格才真正发送抓取坐标”的行为。
# 生产阶段如果要自动发抓取指令，可改成 True。
AUTO_PUBLISH_GRAB_ON_DETECTION = False

# 左侧视角画面中心（光心射线打到世界平面）的世界坐标，单位米
IMAGE_CENTER_WORLD_X_LEFT_M = 0.38
IMAGE_CENTER_WORLD_Y_LEFT_M = 0.02
# 右侧视角画面中心（光心射线打到世界平面）的世界坐标，单位米
IMAGE_CENTER_WORLD_X_RIGHT_M = -0.322
IMAGE_CENTER_WORLD_Y_RIGHT_M = -0.02
# 像素偏移到世界平面坐标的缩放系数，单位 cm/px。
PIXEL_TO_WORLD_SCALE_CM = 0.099
# 发布到 base_footprint 下的固定抓取高度
TARGET_Z_IN_BASE_M = 0.29

# NanoDet 特定配置
STRIDES = [8, 16, 32, 64]
REG_MAX = 7
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32) / 255.0
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32) / 255.0

CLASS_NAMES = [
    "cpepper", "ctomato", "cpumpkin",
    "wonion", "wtomato", "wpumpkin", "wpepper",
]

# 类别索引 -> TARGET_INFO (仿照 picking_detector.py 的测距逻辑)
# 已知宽度(米), 焦距(像素) - 用于 distance = (known_width * focal_length) / pixel_size
TARGET_INFO = {
    0: {'name': 'cpepper', 'known_width': 0.07, 'focal_length': 503.14},
    1: {'name': 'ctomato', 'known_width': 0.09, 'focal_length': 473.33},
    2: {'name': 'cpumpkin', 'known_width': 0.08, 'focal_length': 523.72},
    3: {'name': 'wonion', 'known_width': 0.07, 'focal_length': 393.88},
    4: {'name': 'wtomato', 'known_width': 0.09, 'focal_length': 473.33},
    5: {'name': 'wpumpkin', 'known_width': 0.08, 'focal_length': 523.72},
    6: {'name': 'wpepper', 'known_width': 0.07, 'focal_length': 503.14},
}

K = np.array([
    [507.5, 0, 320.0],
    [0, 507.5, 240.0],
    [0, 0, 1]
 ])

# 手眼标定结果: 相机坐标系 -> 末端坐标系
H_CAMERA_TO_END = np.array([
    [0.952175, 0.04561766, -0.30212877, -0.06881521],
    [-0.28179904, 0.51334522, -0.81059607, 0.09402413],
    [0.11811886, 0.85696891, 0.5016495, 0.03685805],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# ===================== NanoDet 前后处理 =====================

def _nms(boxes, scores, iou_thresh):
    """标准NMS。boxes: (N,4) xyxy，scores: (N,)。"""
    if len(boxes) == 0:
        return []
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
        
    return np.array(keep, dtype=np.intp)


class NanoDetPrePostProcessor:
    """NanoDet模型的预处理和后处理 (来自 picking_detector_onnx.py)"""
    
    def __init__(self, input_h=640, input_w=480, num_classes=7, reg_max=7, strides=[8, 16, 32, 64]):
        self.input_h = input_h
        self.input_w = input_w
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        
        self.center_priors = self._generate_center_priors()
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)
        
    def _generate_center_priors(self):
        """生成所有特征层级的 anchor 中心坐标"""
        priors = []
        for stride in self.strides:
            feat_h = math.ceil(self.input_h / stride)
            feat_w = math.ceil(self.input_w / stride)
            for y in range(feat_h):
                for x in range(feat_w):
                    priors.append([x * stride, y * stride, stride])
        return np.array(priors, dtype=np.float32)
    
    def preprocess(self, img_bgr):
        """预处理图像"""
        orig_h, orig_w = img_bgr.shape[:2]
        
        if img_bgr.shape[:2] != (self.input_h, self.input_w):
            resized = cv2.resize(img_bgr, (self.input_w, self.input_h))
        else:
            resized = img_bgr
        
        blob = resized.astype(np.float32) / 255.0
        blob -= MEAN.reshape(1, 1, 3)
        blob /= STD.reshape(1, 1, 3)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        
        return blob.astype(np.float32), (orig_h, orig_w)
    
    def _decode(self, raw_output):
        """解码 NanoDet 输出"""
        cls_scores = raw_output[:, :self.num_classes]
        reg_preds = raw_output[:, self.num_classes:]
        
        n = reg_preds.shape[0]
        reg_preds = reg_preds.reshape(n, 4, self.reg_max + 1)
        
        reg_preds_max = np.max(reg_preds, axis=-1, keepdims=True)
        reg_preds = np.exp(reg_preds - reg_preds_max)
        reg_preds /= np.sum(reg_preds, axis=-1, keepdims=True)
        
        distances = np.sum(reg_preds * self.project, axis=-1)
        distances *= self.center_priors[:, 2:3]
        
        cx = self.center_priors[:, 0]
        cy = self.center_priors[:, 1]
        x1 = np.clip(cx - distances[:, 0], 0, self.input_w)
        y1 = np.clip(cy - distances[:, 1], 0, self.input_h)
        x2 = np.clip(cx + distances[:, 2], 0, self.input_w)
        y2 = np.clip(cy + distances[:, 3], 0, self.input_h)
        
        boxes = np.stack([x1, y1, x2, y2], axis=-1)
        return boxes, cls_scores
    
    def postprocess(self, raw_output, orig_shape, score_thresh=0.35, nms_thresh=0.6):
        """后处理 - 返回 boxes, scores, class_ids (与 picking_detector 格式一致)"""
        orig_h, orig_w = orig_shape
        
        boxes, cls_scores = self._decode(raw_output)
        
        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y
        
        det_boxes, det_scores, det_labels = [], [], []
        
        for cid in range(self.num_classes):
            scores_c = cls_scores[:, cid]
            mask = scores_c > score_thresh
            
            if not np.any(mask):
                continue
                
            b = boxes[mask]
            s = scores_c[mask]
            
            if len(b) > 0:
                keep = _nms(b, s, nms_thresh)
                if len(keep) > 0:
                    det_boxes.append(b[keep])
                    det_scores.append(s[keep])
                    det_labels.append(np.full(len(keep), cid, dtype=np.int32))
        
        if len(det_boxes) == 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.int32))
        
        return (np.concatenate(det_boxes),
                np.concatenate(det_scores),
                np.concatenate(det_labels))


# ===================== 主节点类 (仿照 picking_detector.py 业务流程) =====================

class PickingDetectorONNX(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.get_logger().info(f'节点初始化完成:{node_name}')

        logger = self.get_logger()
        logger.set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
        self._camera = None
        self._session = None
        self._processor = None
        self._original_h = 0
        self._original_w = 0

        self._fps_total = 0.0
        self._frame_count = 0
        self._display_scale = DISPLAY_SCALE
        self._use_display = USE_DISPLAY
        self._use_low_light_enhance = USE_LOW_LIGHT_ENHANCE
        self._low_light_block_size = LOW_LIGHT_BLOCK_SIZE
        self._low_light_brightness_thresh = LOW_LIGHT_BRIGHTNESS_THRESH
        self._low_light_max_gain = LOW_LIGHT_MAX_GAIN
        
        # 打开相机: 分辨率与模型输入一致可大幅减少预处理时间
        camera_status = self.open_camera(CAMERA_WIDTH, CAMERA_HEIGHT, 30, 0)
        if not camera_status:
            self.get_logger().error("相机初始化失败！")
            return
            
        # 初始化 ONNX 检测模型
        detect_status = self.init_detect(ONNX_MODEL_PATH)
        if not detect_status:
            self.get_logger().error("ONNX模型初始化失败！")
            return

        # 配置相机内参
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.base_frame = 'base_footprint'
        self.end_frame = 'arm_tcp_link4'
        self._debug_dump_requested = False
        self._active_view_name = "left"
        self._active_view_center_x_m = IMAGE_CENTER_WORLD_X_LEFT_M
        self._active_view_center_y_m = IMAGE_CENTER_WORLD_Y_LEFT_M
        self._active_end_x_in_base = 0.0
        self._tf_ready = False
        self._last_tf_warn_time = 0.0
        self._last_need_grab_value = None
        self._last_decision_reason = "init"
        self._qr_detector = cv2.QRCodeDetector()
        self._last_qr_text = None
        self._last_qr_points = None
        self._c_region_label_queue = []
        self._vision_mode = VISION_MODE
        self._work_region = WORK_REGION
        self._pending_recognition = False
        self._frozen_display_frame = None
        self._frozen_display_until = 0.0
        
        self.get_logger().info(f"相机内参: fx={self.fx:.2f}, fy={self.fy:.2f}, 光心坐标: cx={self.cx:.2f}, cy={self.cy:.2f}")
        self.get_logger().info(
            "使用双视角平面映射: left_center=(%.3f, %.3f), right_center=(%.3f, %.3f), pixel_scale=%.6f cm/px" %
            (
                IMAGE_CENTER_WORLD_X_LEFT_M,
                IMAGE_CENTER_WORLD_Y_LEFT_M,
                IMAGE_CENTER_WORLD_X_RIGHT_M,
                IMAGE_CENTER_WORLD_Y_RIGHT_M,
                PIXEL_TO_WORLD_SCALE_CM,
            )
        )
        self.get_logger().info(
            f"当前模式: vision_mode={self._vision_mode}, work_region={self._work_region}, auto_publish={AUTO_PUBLISH_GRAB_ON_DETECTION}"
        )
        self.get_logger().info(
            "低光照增强: enabled=%s block=%s thresh=%.1f max_gain=%.2f"
            % (
                str(self._use_low_light_enhance),
                self._low_light_block_size,
                self._low_light_brightness_thresh,
                self._low_light_max_gain,
            )
        )
        
        # 定时器: 0.016≈60Hz 触发，实际帧率由推理速度决定
        self._timer = self.create_timer(0.016, self.detection_loop)
        
        # 创建发布器
        self._pose_pub = self.create_publisher(PoseStamped, '/arm_pose_cmd', 10)
        self._grab_flag_pub = self.create_publisher(Bool, '/vision_need_grab', 10)
        self._qr_result_pub = self.create_publisher(String, '/qr_code_result', 10)
        self._audio_cmd_pub = self.create_publisher(String, '/audio_cmd', 10)
        
        # 初始化TF监听器
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        
        self.get_logger().info("检测节点已启动，开始检测循环...")

    def _log_detection_result(self, class_name, score, bbox):
        x1, y1, x2, y2 = bbox
        self.get_logger().info(
            "[Detect] target=%s score=%.3f bbox_xyxy=[%.1f, %.1f, %.1f, %.1f]" %
            (class_name, score, x1, y1, x2, y2)
        )

    def _log_image_coordinates(self, pixel_u, pixel_v, rel_x, rel_y):
        self.get_logger().info(
            "[Image2D] pixel=(%.1f, %.1f) center_offset=(%.1f, %.1f) px (origin=image_center, +x=right, +y=down)" %
            (pixel_u, pixel_v, rel_x, rel_y)
        )

    def _log_world_point(self, pixel_u, pixel_v, point_base):
        self.get_logger().info(
            "[World] pixel=(%.1f, %.1f) -> %s=(%.4f, %.4f, %.4f) m" %
            (pixel_u, pixel_v, self.base_frame, point_base[0], point_base[1], point_base[2])
        )

    def _log_pose_publish(self, pose_msg, pixel_u, pixel_v, rel_x, rel_y):
        self.get_logger().info(
            "[Publish] /arm_pose_cmd frame=%s pixel=(%.1f, %.1f) center_offset=(%.1f, %.1f) world=(%.4f, %.4f, %.4f)" %
            (
                pose_msg.header.frame_id,
                pixel_u,
                pixel_v,
                rel_x,
                rel_y,
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z
            )
        )

    def _log_grab_flag_publish(self, flag_value):
        self.get_logger().info(
            "[Publish] /vision_need_grab data=%s" % str(flag_value)
        )

    def _log_qr_result_publish(self, qr_text):
        self.get_logger().info(
            f"[Publish] /qr_code_result data={qr_text}"
        )

    def _log_audio_cmd_publish(self, audio_text):
        self.get_logger().info(
            f"[Publish] /audio_cmd data={audio_text}"
        )

    def _publish_audio_cmd(self, audio_text):
        if not audio_text:
            return
        msg = String()
        msg.data = audio_text
        self._audio_cmd_pub.publish(msg)
        self._log_audio_cmd_publish(audio_text)

    def _log_grab_decision(self, class_name, need_grab, reason):
        self.get_logger().info(
            "[Decision] region=%s mode=%s class=%s need_grab=%s reason=%s" %
            (self._work_region, self._vision_mode, class_name, str(need_grab), reason)
        )

    def _log_onnx_detections(self, boxes, scores, class_ids):
        det_count = len(boxes) if boxes is not None else 0
        self.get_logger().info(f"[ONNX] detection_count={det_count}")

        if boxes is None or scores is None or class_ids is None or det_count == 0:
            self.get_logger().info("[ONNX] no_detection_from_model")
            return

        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            class_idx = int(class_id)
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Unknown_{class_idx}"
            x1, y1, x2, y2 = box
            self.get_logger().info(
                "[ONNX] det[%d] class=%s score=%.3f bbox=[%.1f, %.1f, %.1f, %.1f]" %
                (i, class_name, float(score), x1, y1, x2, y2)
            )

    def _class_name_to_spoken_text(self, class_name):
        if not class_name:
            return "未知目标"

        ripeness_map = {
            "c": "成熟",
            "w": "未成熟",
        }
        fruit_map = {
            "pepper": "辣椒",
            "tomato": "番茄",
            "pumpkin": "南瓜",
            "onion": "洋葱",
        }

        ripeness = ripeness_map.get(class_name[0], "未知")
        fruit_name = fruit_map.get(class_name[1:], class_name[1:])
        return f"{ripeness}{fruit_name}"

    def _build_decision_audio_text(self, class_name, need_grab, reason):
        spoken_name = self._class_name_to_spoken_text(class_name)

        if need_grab:
            if self._work_region == "A":
                return f"识别到{spoken_name}，允许抓取"
            if self._work_region == "C":
                return f"识别到{spoken_name}，匹配队首，允许抓取"
            return f"识别到{spoken_name}，允许抓取"

        if reason == "no_target":
            return "未识别到目标"
        if self._work_region == "A" and reason == "unripe":
            return f"识别到{spoken_name}，未成熟，不抓取"
        if self._work_region == "C" and reason == "not_queue_head":
            return f"识别到{spoken_name}，不是队首，不抓取"
        return f"识别到{spoken_name}，不抓取"

    def _toggle_vision_mode(self):
        self._vision_mode = "onnx" if self._vision_mode == "qr" else "qr"
        self._last_qr_text = None
        self._last_qr_points = None
        self._last_need_grab_value = None
        self.get_logger().info(
            f"切换识别模式: vision_mode={self._vision_mode}, work_region={self._work_region}"
        )
        if self._vision_mode == "onnx":
            self._publish_audio_cmd("切换到目标检测模式")
        else:
            self._publish_audio_cmd("切换到二维码识别模式")

    def _toggle_work_region(self):
        self._work_region = "C" if self._work_region == "A" else "A"
        self._last_need_grab_value = None
        self.get_logger().info(
            f"切换作业区域: work_region={self._work_region}, vision_mode={self._vision_mode}"
        )
        if self._work_region == "A":
            self._publish_audio_cmd("切换到A区")
        else:
            self._publish_audio_cmd("切换到C区")

    def _freeze_display_frame(self, frame, duration_sec=1.0):
        if frame is None:
            return
        self._frozen_display_frame = frame.copy()
        self._frozen_display_until = time.time() + duration_sec

    def _get_display_frame(self, live_frame):
        if (
            self._frozen_display_frame is not None and
            time.time() < self._frozen_display_until
        ):
            return self._frozen_display_frame.copy()

        self._frozen_display_frame = None
        return live_frame

    def _log_active_view(self):
        self.get_logger().info(
            "[View] mode=%s end_x=%.4f center_world=(%.4f, %.4f)" %
            (
                self._active_view_name,
                self._active_end_x_in_base,
                self._active_view_center_x_m,
                self._active_view_center_y_m,
            )
        )

    def _log_coordinate_conversion(self, class_name, score, bbox, pixel_u, pixel_v, offset_x_px,
                                   offset_y_px, delta_world_x_cm, delta_world_y_cm, point_base):
        self._log_active_view()
        self._log_detection_result(class_name, score, bbox)
        self._log_image_coordinates(pixel_u, pixel_v, offset_x_px, offset_y_px)
        self.get_logger().info(
            "[Convert] pixel_offset=(%.1f, %.1f) px -> world_delta=(%.4f, %.4f) cm using scale=%.6f cm/px" %
            (offset_x_px, offset_y_px, delta_world_x_cm, delta_world_y_cm, PIXEL_TO_WORLD_SCALE_CM)
        )
        self._log_world_point(pixel_u, pixel_v, point_base)

    def _publish_need_grab_flag(self, need_grab, force=False, reason=""):
        msg = Bool()
        msg.data = need_grab
        self._grab_flag_pub.publish(msg)
        self._last_need_grab_value = need_grab
        if reason:
            self.get_logger().info(
                "[Publish] /vision_need_grab data=%s reason=%s" % (str(need_grab), reason)
            )
        else:
            self._log_grab_flag_publish(need_grab)

    def _evaluate_grab_need(self, class_name):
        if self._vision_mode != "onnx":
            return False, f"vision_mode={self._vision_mode}"

        if self._work_region == "A":
            if class_name.startswith("c"):
                return True, "mature"
            return False, "unripe"

        if self._work_region == "C":
            current_target = self._get_current_c_region_target_label()
            if current_target is None:
                return False, "no_target"
            if class_name == current_target:
                return True, "queue_head_match"
            return False, "not_queue_head"

        return False, f"unknown_region:{self._work_region}"

    def open_camera(self, width, height, fps, camera_index):
        """打开摄像头"""
        try:
            self._camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

            if not self._camera.isOpened():
                self.get_logger().error(f'相机启动失败,相机编号为:{camera_index}')
                return False
                
            self.get_logger().info(f'相机启动成功,相机编号为{camera_index}')

            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._camera.set(cv2.CAP_PROP_FPS, fps)
            self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲降低延迟
            
            for _ in range(5):
                ret, frame = self._camera.read()
                print(frame.shape)
                if not ret:
                    self.get_logger().warning("相机预热帧读取失败")

            self.get_logger().info(f'相机参数设置完成: {width}x{height}@{fps}fps')
            return True
            
        except Exception as e:
            self.get_logger().error(f"打开相机失败: {e}")
            return False

    def init_detect(self, model_path):
        """初始化 ONNX 推理引擎"""
        try:
            if not os.path.exists(model_path):
                self.get_logger().error(f"模型文件不存在: {model_path}")
                return False
            
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_cpu_mem_arena = False  # Jetson 上可减少内存占用
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 2
            
            providers = []
            if 'TensorrtExecutionProvider' in ort.get_available_providers():
                providers.append(('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_fp16_enable': True,
                }))
                self.get_logger().info("使用 TensorRT 加速 (FP16)")
            elif 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }))
                self.get_logger().info("使用 CUDA 加速")
            else:
                providers.append('CPUExecutionProvider')
                self.get_logger().info("使用 CPU 推理")
            
            self.get_logger().info(f"加载ONNX模型: {model_path}")
            self._session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
            
            # 获取输入输出信息
            inp = self._session.get_inputs()[0]
            out = self._session.get_outputs()[0]
            self.get_logger().info(f"输入名称: {inp.name}, 形状: {inp.shape}")
            self.get_logger().info(f"输出名称: {out.name}, 形状: {out.shape}")
            
            # 解析输出形状得到类别数
            num_classes = out.shape[-1] - 4 * (REG_MAX + 1)
            
            self._processor = NanoDetPrePostProcessor(
                input_h=INPUT_HEIGHT,
                input_w=INPUT_WIDTH,
                num_classes=num_classes,
                reg_max=REG_MAX,
                strides=STRIDES
            )

            # 预热: TensorRT/CUDA 首次推理会较慢，提前跑几次
            try:
                dummy = np.zeros((1, 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32)
                for _ in range(5):
                    self._session.run(None, {self._session.get_inputs()[0].name: dummy})
                self.get_logger().info("推理预热完成")
            except Exception as e:
                self.get_logger().warning(f"预热跳过: {e}")

            self.get_logger().info('ONNX环境初始化完成')
            return True
            
        except Exception as e:
            self.get_logger().error(f'ONNX初始化失败: {e}')
            return False

    def data_infer(self, frame):
        """执行推理 (对应 picking_detector 的 data_infer)"""
        try:
            blob, orig_shape = self._processor.preprocess(frame)
            self._original_h, self._original_w = orig_shape
            
            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: blob})[0]
            
            # NanoDet 输出形状 (N, num_classes + 4*(reg_max+1))
            boxes, scores, class_ids = self._processor.postprocess(
                output[0], orig_shape, score_thresh=0.45, nms_thresh=0.5
            )
            
            # 转为与 YOLOv8 后处理类似的格式，便于后续流程复用
            # picking_detector 的 _postprocess_yolov8_12 返回 boxes_xyxy, confs, class_ids
            # 这里 NanoDet 已经直接返回该格式
            return boxes, scores, class_ids
            
        except Exception as e:
            self.get_logger().error(f"推理失败: {e}")
            return None, None, None

    def _apply_low_light_enhance_if_enabled(self, frame):
        if not self._use_low_light_enhance:
            return frame
        return enhance_frame_bgr_greedy(
            frame,
            block_size=self._low_light_block_size,
            brightness_thresh=self._low_light_brightness_thresh,
            max_gain=self._low_light_max_gain,
        )

    def detect_qr_text(self, frame):
        """识别二维码内容，返回 (text, points)；无结果时返回 (None, None)。"""
        try:
            ok, decoded_text, points, _ = self._qr_detector.detectAndDecodeMulti(frame)
            if ok and decoded_text:
                valid_texts = [text.strip() for text in decoded_text if text and text.strip()]
                if valid_texts:
                    return " | ".join(valid_texts), points

            single_text, single_points, _ = self._qr_detector.detectAndDecode(frame)
            if single_text and single_text.strip():
                return single_text.strip(), single_points

            return None, None
        except Exception as e:
            self.get_logger().warning(f"二维码识别失败: {e}")
            return None, None

    def _log_qr_result(self, qr_text):
        if qr_text is None:
            self.get_logger().info("[QR] no_qr_detected")
        else:
            self.get_logger().info(f"[QR] content={qr_text}")

    def _log_c_region_queue(self):
        self.get_logger().info(f"[CQueue] remaining={self._c_region_label_queue}")

    def _update_c_region_queue_from_qr(self, qr_text):
        """把二维码JSON内容按 key 顺序解析成 C 区抓取队列。"""
        if qr_text is None:
            return False

        try:
            parsed = json.loads(qr_text)
            if not isinstance(parsed, dict):
                self.get_logger().warning("[QR] 二维码内容不是对象字典，无法更新 C 区队列")
                return False

            ordered_items = sorted(parsed.items(), key=lambda item: int(item[0]))
            self._c_region_label_queue = [str(label).strip() for _, label in ordered_items if str(label).strip()]
            self.get_logger().info("[CQueue] 已根据二维码更新抓取顺序")
            self._log_c_region_queue()
            return True
        except Exception as e:
            self.get_logger().warning(f"[QR] 解析二维码顺序失败: {e}")
            return False

    def _get_current_c_region_target_label(self):
        """读取 C 区当前待抓取的第一个 label。"""
        if not self._c_region_label_queue:
            return None
        return self._c_region_label_queue[0]

    def _pop_current_c_region_target_label(self):
        """删除 C 区当前待抓取的第一个 label。"""
        if not self._c_region_label_queue:
            return None
        removed = self._c_region_label_queue.pop(0)
        self.get_logger().info(f"[CQueue] 已完成并移除队首目标: {removed}")
        self._log_c_region_queue()
        return removed

    def draw_qr_result(self, frame, qr_text, qr_points):
        """在画面上绘制二维码框和内容。"""
        if frame is None or qr_text is None:
            return frame

        try:
            if qr_points is not None:
                points = np.array(qr_points, dtype=np.float32)
                points = points.reshape(-1, 2)
                if len(points) >= 4:
                    pts_int = points.astype(np.int32)
                    cv2.polylines(frame, [pts_int], True, (0, 255, 0), 2)
                    text_origin = (int(points[0][0]), max(20, int(points[0][1]) - 10))
                else:
                    text_origin = (20, 60)
            else:
                text_origin = (20, 60)

            cv2.putText(
                frame,
                f"QR:{qr_text}",
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        except Exception as e:
            self.get_logger().warning(f"绘制二维码结果失败: {e}")

        return frame

    def show_detection_window(self, frame, fps=None):
        """显示检测窗口 - 缩放显示以降低开销"""
        try:
            if self._display_scale != 1.0:
                w = int(frame.shape[1] * self._display_scale)
                h = int(frame.shape[0] * self._display_scale)
                resized_frame = cv2.resize(frame, (w, h))
            else:
                resized_frame = frame
            
            if fps is not None:
                fps_text = f"FPS: {fps:.1f}"
                scale = max(0.3, self._display_scale)
                cv2.putText(resized_frame, fps_text, (5, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Detection Results (ONNX)', resized_frame)
            
        except Exception as e:
            self.get_logger().error(f"显示窗口失败: {e}")

    def _update_active_view_from_tf(self):
        """只取末端在世界坐标中的 translation.x 正负切换左/右视角映射。"""
        try:
            transform = self._tf_buffer.lookup_transform(
                self.base_frame,
                self.end_frame,
                rclpy.time.Time()
            )
            end_x = transform.transform.translation.x
            self._active_end_x_in_base = end_x
            if not self._tf_ready:
                self.get_logger().info(
                    f"TF已连接: {self.base_frame} <- {self.end_frame}, 使用 translation.x={end_x:.4f} 判左右"
                )
                self._tf_ready = True

            if end_x >= 0.0:
                self._active_view_name = "left"
                self._active_view_center_x_m = IMAGE_CENTER_WORLD_X_LEFT_M
                self._active_view_center_y_m = IMAGE_CENTER_WORLD_Y_LEFT_M
            else:
                self._active_view_name = "right"
                self._active_view_center_x_m = IMAGE_CENTER_WORLD_X_RIGHT_M
                self._active_view_center_y_m = IMAGE_CENTER_WORLD_Y_RIGHT_M
        except Exception as e:
            now_sec = time.time()
            if now_sec - self._last_tf_warn_time > 2.0:
                self.get_logger().warning(
                    f"获取 {self.base_frame} <- {self.end_frame} TF 失败，暂时沿用当前视角映射: {e}"
                )
                self._last_tf_warn_time = now_sec

    def draw_debug_grid(self, frame, step=40):
        """绘制辅助定位用网格、中心轴和双标尺。"""
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2

        try:
            if self._vision_mode == "qr":
                cv2.putText(
                    frame,
                    f"mode:{self._vision_mode} region:{self._work_region}",
                    (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                return frame

            pixel_tick_color = (40, 40, 40)
            pixel_text_color = (40, 40, 40)
            real_tick_color = (0, 120, 0)
            real_text_color = (0, 120, 0)
            grid_color = (90, 90, 90)
            axis_color = (0, 200, 255)

            # 细网格
            for x in range(0, w, step):
                color = grid_color if x != center_x else axis_color
                thickness = 1 if x != center_x else 2
                cv2.line(frame, (x, 0), (x, h - 1), color, thickness)

            for y in range(0, h, step):
                color = grid_color if y != center_y else axis_color
                thickness = 1 if y != center_y else 2
                cv2.line(frame, (0, y), (w - 1, y), color, thickness)

            # 画面中心十字
            cv2.drawMarker(
                frame,
                (center_x, center_y),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), 2)
            cv2.circle(frame, (center_x, center_y), 2, (255, 255, 255), -1)

            # 中心延长辅助线
            cv2.line(frame, (center_x - 24, center_y), (center_x + 24, center_y), (0, 0, 255), 1)
            cv2.line(frame, (center_x, center_y - 24), (center_x, center_y + 24), (0, 0, 255), 1)

            # 中心点标签
            cv2.putText(
                frame,
                f"Center({center_x},{center_y})",
                (center_x + 10, max(18, center_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )

            # 顶边 X 刻度: 原始像素
            for x in range(0, w, step):
                cv2.line(frame, (x, 0), (x, 10), pixel_tick_color, 2)
                if x < w - 20:
                    cv2.putText(
                        frame,
                        str(x),
                        (x + 2, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        pixel_text_color,
                        1,
                    )

            # 左边 Y 刻度: 原始像素
            for y in range(0, h, step):
                cv2.line(frame, (0, y), (10, y), pixel_tick_color, 2)
                if y < h - 4:
                    cv2.putText(
                        frame,
                        str(y),
                        (10, max(12, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        pixel_text_color,
                        1,
                    )

            # 底边 X 刻度: 像素偏移 * 0.099 后的真实距离
            for x in range(0, w, step):
                offset_x_px = x - center_x
                if self._active_view_name == "left":
                    real_y = -offset_x_px * PIXEL_TO_WORLD_SCALE_CM
                else:
                    real_y = offset_x_px * PIXEL_TO_WORLD_SCALE_CM
                cv2.line(frame, (x, h - 1), (x, h - 11), real_tick_color, 2)
                text = f"{real_y:.2f}"
                text_x = min(max(0, x - 12), w - 34)
                cv2.putText(
                    frame,
                    text,
                    (text_x, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    real_text_color,
                    1,
                )

            # 右边 Y 刻度: 像素偏移 * 0.099 后的真实距离
            for y in range(0, h, step):
                offset_y_px = y - center_y
                if self._active_view_name == "left":
                    real_x = -offset_y_px * PIXEL_TO_WORLD_SCALE_CM
                else:
                    real_x = offset_y_px * PIXEL_TO_WORLD_SCALE_CM
                cv2.line(frame, (w - 1, y), (w - 11, y), real_tick_color, 2)
                text = f"{real_x:.2f}"
                text_y = min(max(10, y + 4), h - 4)
                cv2.putText(
                    frame,
                    text,
                    (max(0, w - 52), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    real_text_color,
                    1,
                )

            cv2.putText(
                frame,
                "top/left:px  bottom/right:*0.099",
                (max(6, w - 230), 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"view:{self._active_view_name} img +x:right +y:down",
                (max(6, w - 245), h - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"center=({self._active_view_center_x_m:.2f},{self._active_view_center_y_m:.2f})",
                (max(6, w - 210), h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"mode:{self._vision_mode} region:{self._work_region}",
                (8, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        except Exception as e:
            self.get_logger().error(f"绘制调试网格失败: {e}")

        return frame

    def calculate_2d_coordinates(self, bbox):
        """
        计算物体在图像中的2D坐标（原点为图像中心）
        
        参数:
            bbox: 边界框 [x1, y1, x2, y2]
        
        返回:
            (relative_x, relative_y): 以图像中心为原点的坐标（像素单位）
        """
        # 1. 计算边界框中心点（图像左上角为原点）
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 2. 图像中心点坐标
        img_center_x = CAMERA_WIDTH / 2
        img_center_y = CAMERA_HEIGHT / 2
        
        # 3. 转换到以图像中心为原点的坐标系
        # X轴: 向右为正
        # Y轴: 向下为正，保持与图像坐标系一致
        relative_x = center_x - img_center_x
        relative_y = center_y - img_center_y
        
        return relative_x, relative_y

    def _get_bbox_center_pixel(self, bbox):
        """返回检测框中心像素坐标，图像左上角为原点。"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        return center_x, center_y

    def _pixel_offset_to_world(self, offset_x_px, offset_y_px):
        """
        根据图像中心偏移直接映射到世界平面坐标。
        图像系: +x 右, +y 下
        左视角:
        - 图像 x 偏移对应世界 -y
        - 图像 y 偏移对应世界 -x
        右视角:
        - 图像 x 偏移对应世界 +y
        - 图像 y 偏移对应世界 +x
        """
        if self._active_view_name == "left":
            delta_world_x_cm = -offset_y_px * PIXEL_TO_WORLD_SCALE_CM
            delta_world_y_cm = -offset_x_px * PIXEL_TO_WORLD_SCALE_CM
        else:
            delta_world_x_cm = offset_y_px * PIXEL_TO_WORLD_SCALE_CM
            delta_world_y_cm = offset_x_px * PIXEL_TO_WORLD_SCALE_CM
        delta_world_x_m = delta_world_x_cm / 100.0
        delta_world_y_m = delta_world_y_cm / 100.0

        point_base = np.array([
            self._active_view_center_x_m + delta_world_x_m,
            self._active_view_center_y_m + delta_world_y_m,
            0.0,
        ], dtype=np.float64)

        return point_base, delta_world_x_cm, delta_world_y_cm



    def detection_loop(self):
        """主检测循环 (与 picking_detector 业务流程一致)"""
        try:
            ret, frame = self._camera.read()
            if not ret:
                self.get_logger().warning("无法从相机读取帧")
                return

            frame = self._apply_low_light_enhance_if_enabled(frame)

            start_time = time.time()
            self._update_active_view_from_tf()
            display_frame = self.draw_debug_grid(frame.copy())
            display_frame = self._get_display_frame(display_frame)

            end_time = time.time()
            fps = 1.0 / (end_time - start_time) if end_time > start_time else 0
            self._fps_total += fps
            self._frame_count += 1

            if self._use_display:
                self.show_detection_window(display_frame, fps)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1

            if key == ord('c') or key == ord('C'):
                self._toggle_vision_mode()
            if key == ord('d') or key == ord('D'):
                self._toggle_work_region()

            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)
            class_ids = np.zeros(0, dtype=np.int32)
            class_name = None
            need_grab = False

            if key == 32:
                self.get_logger().info(f"触发一次识别: vision_mode={self._vision_mode}")

                if self._vision_mode == "onnx":
                    result = self.data_infer(frame)
                    if result[0] is None:
                        return

                    boxes, scores, class_ids = result
                    self._log_onnx_detections(boxes, scores, class_ids)
                    if len(boxes) > 0:
                        display_frame = self.draw_annotation(display_frame, boxes, scores, class_ids)
                        self._freeze_display_frame(display_frame, duration_sec=1.0)
                        target_idx = int(np.argmax(scores))
                        class_id = class_ids[target_idx]
                        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Unknown_{class_id}"
                        need_grab, decision_reason = self._evaluate_grab_need(class_name)
                        self._last_decision_reason = decision_reason
                        self._log_grab_decision(class_name, need_grab, decision_reason)
                        self._publish_audio_cmd(
                            self._build_decision_audio_text(class_name, need_grab, decision_reason)
                        )

                        if need_grab:
                            annotated_frame = self.draw_annotation(frame.copy(), boxes, scores, class_ids)
                            self._debug_dump_requested = True
                            self.publish_detection_results(annotated_frame, boxes, scores, class_ids)
                            self._debug_dump_requested = False
                            self.get_logger().info(f"=== 已发送抓取指令: {class_name} ===")
                        else:
                            self._publish_need_grab_flag(False, reason=decision_reason)
                    else:
                        self._last_decision_reason = "no_target"
                        self._publish_need_grab_flag(False, reason="no_target")
                        self._publish_audio_cmd("未识别到目标")

                elif self._vision_mode == "qr":
                    qr_text, qr_points = self.detect_qr_text(frame)
                    if qr_text is not None:
                        self._log_qr_result(qr_text)
                        self._last_qr_text = qr_text
                        self._update_c_region_queue_from_qr(qr_text)
                        qr_msg = String()
                        qr_msg.data = qr_text
                        self._qr_result_pub.publish(qr_msg)
                        self._log_qr_result_publish(qr_text)
                        self._publish_audio_cmd("二维码识别成功")
                    else:
                        self._log_qr_result(None)
                        self._last_qr_text = None
                        self._publish_audio_cmd("未识别到二维码")
                    self._last_qr_points = qr_points
                    display_frame = self.draw_qr_result(display_frame, qr_text, qr_points)
                    self._freeze_display_frame(display_frame, duration_sec=1.0)

                if self._use_display:
                    self.show_detection_window(display_frame, fps)

            if key == 27:
                self.get_logger().info("用户按ESC退出")
                self.destroy_node()
                return
            
        except Exception as e:
            self.get_logger().error(f"检测循环错误: {e}")

        
    def _create_results_from_detections(self, boxes, scores, class_ids, frame):
        """创建结果对象"""
        if len(boxes) == 0:
            return []
        
        try:
            result = Result(
                boxes_data=boxes,
                scores_data=scores,
                class_ids_data=class_ids,
                orig_shape=frame.shape[:2],
                orig_img=frame
            )
            return [result]
            
        except Exception as e:
            self.get_logger().error(f"创建结果对象失败: {e}")
            return []

    def distance_calculate(self, results):
        """计算距离 (与 picking_detector 一致)"""
        distance_list = []

        if not results:
            return distance_list
        
        try:
            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                    
                boxes_data = getattr(result.boxes, 'data', None)
                if boxes_data is None or len(boxes_data) == 0:
                    continue

                for i in range(len(boxes_data)):
                    try:
                        x1, y1, x2, y2, score, cls = boxes_data[i]
                        cls_int = int(cls)

                        if cls_int not in TARGET_INFO:
                            continue

                        info = TARGET_INFO[cls_int]
                        
                        if info['known_width'] == 0 or info['focal_length'] == 0:
                            continue

                        pixel_width = float(x2 - x1)
                        pixel_height = float(y2 - y1)
                        # 用宽度进行测距 (known_width 对应宽度)
                        pixel_size = pixel_width

                        if pixel_size <= 0:
                            continue

                        # 统一使用相机fx作为焦距 (避免不同类别的标定误差)
                        focal_length = self.fx  # 403.1075
                        distance = (info['known_width'] * focal_length) / pixel_size

                        distance_list.append({
                            'cls': cls_int,
                            'distance': float(distance)
                        })

                    except Exception as e:
                        self.get_logger().warning(f"单个目标测距异常: {e}")
                        continue
                        
        except Exception as e:
            self.get_logger().error(f"测距计算失败: {e}")
            
        return distance_list

    def draw_annotation(self, frame, boxes, scores, class_ids):
        """绘制检测框和标签 - 保留原有功能，增加2D坐标显示"""
        if boxes is None or len(boxes) == 0:
            return frame
            
        try:
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                if len(box) != 4:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Unknown_{class_id}"
                label = f"{class_name}: {score:.2f}"
                
                # ========== 新增：计算2D坐标 ==========
                rel_x, rel_y = self.calculate_2d_coordinates(box)
                pixel_u, pixel_v = self._get_bbox_center_pixel(box)
                coord_label = f"2D:({rel_x:.0f},{rel_y:.0f})"
                pixel_label = f"px:({pixel_u:.0f},{pixel_v:.0f})"
                # ========== 新增结束 ==========
                
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (int(pixel_u), int(pixel_v)), 4, (0, 0, 255), -1)
                cv2.drawMarker(
                    frame,
                    (int(pixel_u), int(pixel_v)),
                    (255, 255, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=1,
                )
                
                # 绘制类别标签
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_y = max(y1 - 5, 0)
                cv2.rectangle(frame, (x1, label_y - label_h - 5), 
                            (x1 + label_w, label_y), color, -1)
                cv2.putText(frame, label, (x1, label_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # ========== 新增：绘制2D坐标标签 ==========
                # 在边界框下方显示2D坐标
                coord_y = y2 + 20
                (coord_w, coord_h), _ = cv2.getTextSize(coord_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                if coord_y + coord_h < frame.shape[0]:  # 确保不超出图像边界
                    cv2.rectangle(frame, (x1, coord_y - coord_h - 3),
                                (x1 + coord_w, coord_y + 3), (255, 100, 100), -1)
                    cv2.putText(frame, coord_label, (x1, coord_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                pixel_text_y = min(frame.shape[0] - 5, coord_y + 16)
                (pixel_w, pixel_h), _ = cv2.getTextSize(pixel_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, pixel_text_y - pixel_h - 3),
                            (x1 + pixel_w, pixel_text_y + 3), (80, 180, 255), -1)
                cv2.putText(frame, pixel_label, (x1, pixel_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                # ========== 新增结束 ==========
                    
        except Exception as e:
            self.get_logger().error(f"绘制标注失败: {e}")
            
        return frame




    def publish_detection_results(self, frame, boxes, scores, class_ids):
        """发布检测结果。使用图像中心参考点和像素偏移直接映射到世界平面坐标。"""
        try:
            if len(boxes) > 0 and len(scores) > 0 and len(class_ids) > 0:
                target_idx = int(np.argmax(scores))

                if target_idx < len(boxes):
                    box = boxes[target_idx]
                    pixel_u, pixel_v = self._get_bbox_center_pixel(box)
                    offset_x_px, offset_y_px = self.calculate_2d_coordinates(box)
                    point_base, delta_world_x, delta_world_y = self._pixel_offset_to_world(
                        offset_x_px, offset_y_px
                    )
                    class_id = int(class_ids[target_idx])
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Unknown_{class_id}"

                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = self.base_frame
                    pose_msg.pose.position.x = float(point_base[0]) - 0.03
                    self.get_logger().warning(f"调试x: {pose_msg.pose.position.x}, 原始x: {float(point_base[0])}")

                    pose_msg.pose.position.y = float(point_base[1]) - 0.01
                    self.get_logger().warning(f"调试y: {pose_msg.pose.position.y}, 原始y: {float(point_base[1])}")

                    pose_msg.pose.position.z = TARGET_Z_IN_BASE_M
                    pose_msg.pose.orientation.x = 0.0
                    pose_msg.pose.orientation.y = 0.0
                    pose_msg.pose.orientation.z = -0.7071067811865475
                    pose_msg.pose.orientation.w = 0.7071067811865476

                    grab_msg = Bool()
                    grab_msg.data = True
                    self._grab_flag_pub.publish(grab_msg)
                    self._log_grab_flag_publish(grab_msg.data)
                    self._publish_audio_cmd(f"开始抓取{self._class_name_to_spoken_text(class_name)}")
                    time.sleep(1)
                    self._pose_pub.publish(pose_msg)
                    self._log_pose_publish(pose_msg, pixel_u, pixel_v, offset_x_px, offset_y_px)
                    if self._work_region == "C":
                        self._pop_current_c_region_target_label()

                    if self._debug_dump_requested:
                        self._log_coordinate_conversion(
                            class_name,
                            float(scores[target_idx]),
                            box,
                            pixel_u,
                            pixel_v,
                            offset_x_px,
                            offset_y_px,
                            delta_world_x,
                            delta_world_y,
                            point_base
                        )
                        self._debug_dump_requested = False
        except Exception as e:
            self.get_logger().warning(f"发布结果失败: {e}")


    def cleanup(self):
        """清理资源"""
        try:
            self.get_logger().info("开始清理资源...")
            
            if self._frame_count > 0:
                avg_fps = self._fps_total / self._frame_count
                self.get_logger().info(f"平均帧率: {avg_fps:.2f} FPS")

            if self._camera is not None:
                self._camera.release()
                self._camera = None
                self.get_logger().info("相机资源已释放")
                
            self._session = None
            cv2.destroyAllWindows()
            self.get_logger().info("资源清理完成")
            
        except Exception as e:
            self.get_logger().error(f"资源清理失败: {e}")

    def destroy_node(self):
        self.cleanup()
        super().destroy_node()

    def __del__(self):
        self.cleanup()


# ===================== Result/Boxes/Box 类 (与 picking_detector 一致) =====================

class Result:
    def __init__(self, boxes_data, scores_data, class_ids_data, orig_shape, orig_img=None):
        if len(boxes_data) > 0:
            data = np.column_stack([
                boxes_data, 
                scores_data[:, None], 
                class_ids_data[:, None]
            ])
        else:
            data = np.empty((0, 6))
        
        self.boxes = Boxes(data, orig_shape)
        self.orig_shape = orig_shape
        self.orig_img = orig_img
    
    def __iter__(self):
        for box in self.boxes:
            yield box
    
    def __len__(self):
        return len(self.boxes)
    
    def __str__(self):
        if len(self.boxes) == 0:
            return "Result(无检测)"
        
        result_str = f"Result(nums:{len(self.boxes)})"
        
        for i, box in enumerate(self.boxes):
            try:
                xyxy = box.xyxy
                if len(xyxy.shape) == 2:
                    x1, y1, x2, y2 = xyxy[0]
                else:
                    x1, y1, x2, y2 = xyxy[:4]
                
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                result_str += f"detect{i}: cls={cls_id}, conf={conf:.3f}, center=({center_x:.1f},{center_y:.1f}), box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                
            except:
                result_str += f"检测{i}: 格式错误"
        
        return result_str
    
    def __repr__(self):
        return self.__str__()
    
class Boxes:
    def __init__(self, data, orig_shape):
        self.data = data
        self.orig_shape = orig_shape
        
    def __iter__(self):
        for i in range(len(self.data)):
            yield Box(self.data[i])
    
    def __len__(self):
        return len(self.data)
    
class Box:
    def __init__(self, data):
        self.data = data
    
    @property
    def xyxy(self):
        return self.data[:4].reshape(1, -1)
    
    @property
    def xywh(self):
        x1, y1, x2, y2 = self.data[:4]
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    
    @property
    def cls(self):
        return self.data[5:6].reshape(1, -1)
    
    @property
    def conf(self):
        return self.data[4:5].reshape(1, -1)
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self.data
    
def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = PickingDetectorONNX("picking_detector_onnx_ai")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        if node is not None:
            node.get_logger().info("程序被用户中断")
    except Exception as e:
        if node is not None:
            node.get_logger().error(f"程序运行错误: {e}")
        else:
            print(f"程序运行错误: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
