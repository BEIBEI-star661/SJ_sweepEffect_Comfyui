import torch
import numpy as np
from PIL import Image
import nodes
import math
import torch.nn.functional as F

class SJ_sweepEffect:
    CATEGORY = "CSJ"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sweep_speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "sweep_intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "sweep_angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "blur_radius": ("INT", {"default": 30, "min": 0, "max": 100, "step": 20}),
                "sweep_width": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 0.5, "step": 0.05}),
                "sweep_color": ("COLOR", {"default": "#FFFFFF"}),
                "pure_color_mode": ("BOOLEAN", {"default": False, "label_on": "Mixed Color", "label_off": "White"}),
                "frames": ("INT", {"default": 24, "min": 16, "max": 30, "step": 1}),
                "quality": (["low", "medium", "high"], {"default": "medium"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "apply_sweep_effect"

    def separable_gaussian_blur(self, tensor, kernel_size, sigma):
        """
        使用可分离高斯核进行更高效的模糊处理
        """
        # 确保kernel_size是奇数整数
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 最小核大小为3
        kernel_size = max(kernel_size, 3)
        
        # 创建1D高斯核
        coords = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device)
        coords = coords - kernel_size // 2
        g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        g1d = g1d.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # 确保padding是整数
        padding = int(kernel_size // 2)
        
        # 水平模糊
        tensor_expanded = tensor.unsqueeze(0).unsqueeze(0)
        tensor_padded = F.pad(tensor_expanded, (padding, padding, 0, 0), mode='reflect')
        blurred_h = F.conv2d(tensor_padded, g1d, padding=0)
        
        # 垂直模糊
        g1d_v = g1d.transpose(-1, -2)
        tensor_padded = F.pad(blurred_h, (0, 0, padding, padding), mode='reflect')
        blurred_v = F.conv2d(tensor_padded, g1d_v, padding=0)
        
        return blurred_v.squeeze(0).squeeze(0)

    def create_super_antialiased_mask(self, distance_from_sweep, sweep_width_pixels, quality_level, device):
        """
        创建超级抗锯齿的扫光遮罩
        """
        # 提升抗锯齿质量
        if quality_level == "high":
            aa_samples = 25  # 5x5 超采样
            edge_samples = 9  # 3x3 边缘采样
            blur_kernel = 9
            blur_sigma = 1.8
        elif quality_level == "medium":
            aa_samples = 9   # 3x3 超采样
            edge_samples = 4  # 2x2 边缘采样
            blur_kernel = 7
            blur_sigma = 1.2
        else:
            aa_samples = 4   # 2x2 超采样
            edge_samples = 1  # 单采样
            blur_kernel = 5
            blur_sigma = 0.8
        
        # 主要抗锯齿采样
        sqrt_samples = int(math.sqrt(aa_samples))
        offsets = torch.linspace(-0.5, 0.5, sqrt_samples + 2)[1:-1]
        
        mask_accumulator = torch.zeros_like(distance_from_sweep)
        
        for dy in offsets:
            for dx in offsets:
                # 子像素偏移
                offset_distance = distance_from_sweep + dx * 0.7 + dy * 0.7
                
                # 计算子像素遮罩
                if sweep_width_pixels > 0:
                    # 核心扫光区域 - 更锐利
                    core_sigma = sweep_width_pixels * 0.4
                    core_mask = torch.exp(-0.5 * (offset_distance / core_sigma) ** 2)
                    
                    # 中间过渡区域
                    mid_sigma = sweep_width_pixels * 0.8
                    mid_mask = torch.exp(-0.5 * (offset_distance / mid_sigma) ** 2) * 0.6
                    
                    # 柔和边缘区域
                    edge_sigma = sweep_width_pixels * 1.4
                    edge_mask = torch.exp(-0.5 * (offset_distance / edge_sigma) ** 2) * 0.2
                    
                    # 组合多层遮罩
                    combined_mask = torch.maximum(torch.maximum(core_mask, mid_mask), edge_mask)
                else:
                    combined_mask = torch.zeros_like(offset_distance)
                
                mask_accumulator += combined_mask
        
        # 平均所有子像素样本
        sweep_mask = mask_accumulator / aa_samples
        
        # 额外的边缘平滑处理
        if edge_samples > 1:
            edge_sqrt = int(math.sqrt(edge_samples))
            edge_offsets = torch.linspace(-0.25, 0.25, edge_sqrt + 1)[:-1] + 0.125
            
            edge_accumulator = torch.zeros_like(distance_from_sweep)
            for edy in edge_offsets:
                for edx in edge_offsets:
                    edge_distance = distance_from_sweep + edx + edy
                    if sweep_width_pixels > 0:
                        edge_sigma = sweep_width_pixels * 1.8
                        edge_smooth = torch.exp(-0.5 * (edge_distance / edge_sigma) ** 2) * 0.15
                        edge_accumulator += edge_smooth
            
            edge_mask = edge_accumulator / edge_samples
            sweep_mask = torch.maximum(sweep_mask, edge_mask)
        
        # 应用可分离高斯模糊进一步抗锯齿
        if blur_kernel >= 5:
            sweep_mask = self.separable_gaussian_blur(sweep_mask, blur_kernel, blur_sigma)
        
        return sweep_mask

    def apply_sweep_effect(self, image, sweep_speed, sweep_intensity, sweep_angle, blur_radius, sweep_width, sweep_color, pure_color_mode, frames, quality):
        # 设备和数据类型处理
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32))
        
        original_device = image.device
        if original_device.type == 'cpu' and torch.cuda.is_available():
            device = torch.device('cuda')
            image = image.to(device)
        else:
            device = original_device
        
        image = image.to(torch.float32)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        # 处理批次维度
        if len(image.shape) == 4:
            image = image.squeeze(0)
        
        height, width, channels = image.shape

        # 解析扫光颜色 - 确保颜色只影响扫光区域
        if pure_color_mode:
            color_hex = sweep_color.lstrip('#')
            if len(color_hex) == 6:
                # 正确解析RGB颜色
                color = torch.tensor([
                    int(color_hex[0:2], 16) / 255.0,
                    int(color_hex[2:4], 16) / 255.0,
                    int(color_hex[4:6], 16) / 255.0
                ], dtype=torch.float32, device=device)
            else:
                color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        else:
            # 默认白色扫光
            color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

        # 生成帧序列
        frame_sequence = []
        
        for frame_idx in range(frames):
            # 计算扫光进度
            base_progress = frame_idx / max(frames - 1, 1)
            total_sweep_duration = 1.0 / sweep_speed
            adjusted_progress = min(base_progress / total_sweep_duration, 1.0)
            
            # 使用三次Hermite插值函数，比smoothstep更平滑
            t = adjusted_progress
            smooth_progress = t * t * t * (t * (t * 6 - 15) + 10)  # smootherstep
            
            # 创建高精度坐标网格
            y_coords, x_coords = torch.meshgrid(
                torch.arange(height, dtype=torch.float32, device=device),
                torch.arange(width, dtype=torch.float32, device=device),
                indexing='ij'
            )
            
            # 应用旋转（如果需要）
            if abs(sweep_angle) > 0.1:
                angle_rad = math.radians(sweep_angle)
                center_x = width / 2.0
                center_y = height / 2.0
                
                x_centered = x_coords - center_x
                y_centered = y_coords - center_y
                
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                
                x_rotated = x_centered * cos_a - y_centered * sin_a
                y_rotated = x_centered * sin_a + y_centered * cos_a
                
                x_coords = x_rotated + center_x
                y_coords = y_rotated + center_y
            
            # 计算扫光位置
            diagonal_length = math.sqrt(height**2 + width**2)
            sweep_start = -diagonal_length * 0.3
            sweep_end = diagonal_length * 1.3
            sweep_position = sweep_start + smooth_progress * (sweep_end - sweep_start)
            
            # 计算到扫光线的距离（提高精度）
            distance_from_sweep = torch.abs(y_coords + x_coords - sweep_position) / math.sqrt(2)
            
            # 创建超级抗锯齿扫光遮罩
            sweep_width_pixels = sweep_width * min(height, width)
            sweep_mask = self.create_super_antialiased_mask(distance_from_sweep, sweep_width_pixels, quality, device)
            
            # 应用额外的模糊效果 - 修复整数转换问题
            if blur_radius > 0:
                blur_sigma = blur_radius / 2.5
                # 确保blur_kernel_size是整数
                blur_kernel_size = int(min(blur_radius // 1.5, 21))
                if blur_kernel_size % 2 == 0:
                    blur_kernel_size += 1
                # 最小核大小
                blur_kernel_size = max(blur_kernel_size, 3)
                
                blur_distance = distance_from_sweep / blur_sigma
                blur_mask = torch.exp(-0.5 * blur_distance ** 2) * 0.2
                
                # 对模糊遮罩应用可分离高斯模糊
                if blur_kernel_size >= 5:
                    blur_mask = self.separable_gaussian_blur(blur_mask, blur_kernel_size, blur_sigma * 0.4)
                
                sweep_mask = torch.maximum(sweep_mask, blur_mask)
            
            # 应用强度
            sweep_mask = sweep_mask * sweep_intensity
            
            # 使用更平滑的限制函数
            sweep_mask = torch.sigmoid(sweep_mask * 4.0 - 2.0)  # 更平滑的S曲线
            sweep_mask = torch.clamp(sweep_mask, 0.0, 1.0)
            
            # 最终抗锯齿处理
            if quality in ["medium", "high"]:
                final_blur_sigma = 0.4 if quality == "medium" else 0.6
                final_kernel = 5 if quality == "medium" else 7
                sweep_mask = self.separable_gaussian_blur(sweep_mask, final_kernel, final_blur_sigma)
            
            # 创建当前帧 - 保持原图完整性
            current_frame = image.clone()
            
            # 应用扫光效果 - 颜色只影响扫光区域
            light_effect = sweep_mask.unsqueeze(-1) * color
            alpha = sweep_mask.unsqueeze(-1)
            
            # 使用屏幕混合模式获得更自然的光效
            # Screen模式: result = 1 - (1 - base) * (1 - blend)
            inv_base = 1.0 - current_frame
            inv_light = 1.0 - light_effect
            screen_result = 1.0 - (inv_base * inv_light)
            
            # 在扫光区域应用屏幕混合，其他区域保持原样
            current_frame = current_frame * (1.0 - alpha) + screen_result * alpha
            
            # 确保值在有效范围内
            current_frame = torch.clamp(current_frame, 0.0, 1.0)
            
            frame_sequence.append(current_frame)
        
        # 堆叠帧序列
        frame_sequence = torch.stack(frame_sequence, dim=0)
        frame_sequence = frame_sequence.contiguous()
        frame_sequence = frame_sequence.to(torch.float32)
        frame_sequence = frame_sequence.to(device)
        frame_sequence = torch.clamp(frame_sequence, 0.0, 1.0)
        
        return (frame_sequence,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "SJ_sweepEffect": SJ_sweepEffect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SJ_sweepEffect": "SJ Sweep Effect",
}