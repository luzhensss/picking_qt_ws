# -*- coding: utf-8 -*-
"""
低光照图像增强（贪心分块、关键区域由“更暗块优先增强”体现）。

设计依据为《基于贪心思想的低光照图像的数据增强》思路（与桌面项目
「低光照图像增强（关键区域优先）」选题说明一致；原仓库 C++ 文件为占位，
此处为可部署的 Python/OpenCV 实现）：

- 局部最优：每个块单独根据本块平均亮度决定是否增强及增强强度；
- 无回溯：按网格从左到右、从上到下处理，处理完即写入输出；
- 阈值贪心：块平均灰度 >= brightness_thresh 的块不增强，否则按系数抬升 V 通道；
- 分块 ROI：与文档中 Rect(x,y,blockSize,blockSize) 切片思路一致，图像边缘块可能小于 blockSize。
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def enhance_frame_bgr_greedy(
    image_bgr: np.ndarray,
    block_size: int = 32,
    brightness_thresh: float = 80.0,
    max_gain: float = 2.5,
) -> np.ndarray:
    """
    对 BGR uint8 图像做贪心分块低光照增强，返回新图（不修改输入）。

    :param image_bgr: BGR，uint8
    :param block_size: 窗口边长（与文档 blockSize 对应）
    :param brightness_thresh: 平均灰度阈值，大于等于则不增强（文档取 80）
    :param max_gain: 对 V 通道乘法的上限，抑制过曝与噪声放大
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    if image_bgr.dtype != np.uint8:
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    h, w = image_bgr.shape[:2]
    if block_size < 4:
        block_size = 4

    out = image_bgr.copy()
    for y in range(0, h, block_size):
        y2 = min(y + block_size, h)
        for x in range(0, w, block_size):
            x2 = min(x + block_size, w)
            block = image_bgr[y:y2, x:x2]
            mean_gray = float(np.mean(cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)))
            if mean_gray >= brightness_thresh:
                continue

            # 越暗的块 gain 越大，上限 max_gain（短视、仅本块统计）
            gain = min(max_gain, brightness_thresh / max(mean_gray, 1e-3))
            hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
            hh, ss, vv = cv2.split(hsv)
            vv_f = vv.astype(np.float32) * gain
            vv = np.clip(vv_f, 0, 255).astype(np.uint8)
            merged = cv2.merge([hh, ss, vv])
            out[y:y2, x:x2] = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

    return out


def block_mean_gray_bgr(block_bgr: np.ndarray) -> float:
    """单块平均灰度，供测试或外部调试。"""
    return float(np.mean(cv2.cvtColor(block_bgr, cv2.COLOR_BGR2GRAY)))


def greedy_gain(mean_gray: float, brightness_thresh: float, max_gain: float) -> Tuple[bool, float]:
    """是否增强及增益（与 enhance_frame_bgr_greedy 中规则一致）。"""
    if mean_gray >= brightness_thresh:
        return False, 1.0
    g = min(max_gain, brightness_thresh / max(mean_gray, 1e-3))
    return True, g
