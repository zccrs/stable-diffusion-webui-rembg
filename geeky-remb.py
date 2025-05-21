import os
import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageColor
import cv2
from tqdm import tqdm
import gradio as gr
import torch
import tempfile
from concurrent.futures import ThreadPoolExecutor
import queue
from threading import Thread

import modules.scripts as scripts
from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
from modules.paths_internal import models_path

class BlendMode:
    @staticmethod
    def _ensure_same_channels(target, blend):
        """Ensure both images have the same number of channels"""
        if target.shape[-1] == 4 and blend.shape[-1] == 3:
            alpha = np.ones((*blend.shape[:2], 1))
            blend = np.concatenate([blend, alpha], axis=-1)
        elif target.shape[-1] == 3 and blend.shape[-1] == 4:
            alpha = np.ones((*target.shape[:2], 1))
            target = np.concatenate([target, alpha], axis=-1)
        return target, blend

    @staticmethod
    def _apply_blend(target, blend, operation, opacity=1.0):
        """Apply blend operation with proper alpha handling"""
        target, blend = BlendMode._ensure_same_channels(target, blend)

        target_rgb = target[..., :3]
        blend_rgb = blend[..., :3]

        target_a = target[..., 3:] if target.shape[-1] == 4 else 1
        blend_a = blend[..., 3:] if blend.shape[-1] == 4 else 1

        result_rgb = operation(target_rgb, blend_rgb)
        result_a = target_a * blend_a

        result_rgb = result_rgb * opacity + target_rgb * (1 - opacity)
        result_a = result_a * opacity + target_a * (1 - opacity)

        return np.concatenate([result_rgb, result_a], axis=-1) if target.shape[-1] == 4 else result_rgb

    @staticmethod
    def normal(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: b, opacity)

    @staticmethod
    def multiply(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: t * b, opacity)

    @staticmethod
    def screen(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: 1 - (1 - t) * (1 - b), opacity)

    @staticmethod
    def overlay(target, blend, opacity=1.0):
        def overlay_op(t, b):
            return np.where(t > 0.5,
                          1 - 2 * (1 - t) * (1 - b),
                          2 * t * b)
        return BlendMode._apply_blend(target, blend, overlay_op, opacity)

    @staticmethod
    def soft_light(target, blend, opacity=1.0):
        def soft_light_op(t, b):
            return np.where(b > 0.5,
                          t + (2 * b - 1) * (t - t * t),
                          t - (1 - 2 * b) * t * (1 - t))
        return BlendMode._apply_blend(target, blend, soft_light_op, opacity)

    @staticmethod
    def hard_light(target, blend, opacity=1.0):
        def hard_light_op(t, b):
            return np.where(b > 0.5,
                          1 - (1 - t) * (2 - 2 * b),
                          2 * t * b)
        return BlendMode._apply_blend(target, blend, hard_light_op, opacity)

    @staticmethod
    def difference(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: np.abs(t - b), opacity)

    @staticmethod
    def exclusion(target, blend, opacity=1.0):
        return BlendMode._apply_blend(target, blend, lambda t, b: t + b - 2 * t * b, opacity)

    @staticmethod
    def color_dodge(target, blend, opacity=1.0):
        def color_dodge_op(t, b):
            return np.where(b >= 1, 1, np.minimum(1, t / (1 - b + 1e-6)))
        return BlendMode._apply_blend(target, blend, color_dodge_op, opacity)

    @staticmethod
    def color_burn(target, blend, opacity=1.0):
        def color_burn_op(t, b):
            return np.where(b <= 0, 0, np.maximum(0, 1 - (1 - t) / (b + 1e-6)))
        return BlendMode._apply_blend(target, blend, color_burn_op, opacity)

class GeekyRemB:
    def __init__(self):
        self.session = None
        if "U2NET_HOME" not in os.environ:
            os.environ["U2NET_HOME"] = os.path.join(models_path, "u2net")
        self.use_gpu = torch.cuda.is_available()
        self.blend_mode_map = {
            "正常": "normal",
            "正片叠底": "multiply",
            "滤色": "screen",
            "叠加": "overlay",
            "柔光": "soft_light",
            "强光": "hard_light",
            "差值": "difference",
            "排除": "exclusion",
            "颜色减淡": "color_dodge",
            "颜色加深": "color_burn"
        }
        self.background_mode_map = {
            "透明": "transparent",
            "纯色": "color",
            "图片": "image"
        }
        self.chroma_key_map = {
            "无": "none",
            "绿色": "green",
            "蓝色": "blue",
            "红色": "red"
        }
        self.blend_modes = {
            "normal": BlendMode.normal,
            "multiply": BlendMode.multiply,
            "screen": BlendMode.screen,
            "overlay": BlendMode.overlay,
            "soft_light": BlendMode.soft_light,
            "hard_light": BlendMode.hard_light,
            "difference": BlendMode.difference,
            "exclusion": BlendMode.exclusion,
            "color_dodge": BlendMode.color_dodge,
            "color_burn": BlendMode.color_burn
        }

    def apply_blend_mode(self, target, blend, mode="normal", opacity=1.0):
        if mode not in self.blend_modes:
            return blend

        target = target.astype(np.float32) / 255
        blend = blend.astype(np.float32) / 255

        result = self.blend_modes[mode](target, blend, opacity)

        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def apply_chroma_key(self, image, color, threshold, color_tolerance=20):
        if isinstance(image, Image.Image):
            image = np.array(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if color == "green":
            lower = np.array([40 - color_tolerance, 40, 40])
            upper = np.array([80 + color_tolerance, 255, 255])
        elif color == "blue":
            lower = np.array([90 - color_tolerance, 40, 40])
            upper = np.array([130 + color_tolerance, 255, 255])
        elif color == "red":
            lower = np.array([0, 40, 40])
            upper = np.array([20 + color_tolerance, 255, 255])
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = 255 - cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
        return mask

    def process_mask(self, mask, invert_mask, feather_amount, mask_blur, mask_expansion):
        if invert_mask:
            mask = 255 - mask

        if mask_expansion != 0:
            kernel = np.ones((abs(mask_expansion), abs(mask_expansion)), np.uint8)
            if mask_expansion > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)

        if feather_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_amount)

        if mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur)

        return mask

    def parse_aspect_ratio(self, aspect_ratio_input):
        if not aspect_ratio_input:
            return None

        if ':' in aspect_ratio_input:
            try:
                w, h = map(float, aspect_ratio_input.split(':'))
                return w / h
            except ValueError:
                return None

        try:
            return float(aspect_ratio_input)
        except ValueError:
            pass

        standard_ratios = {
            '4:3': 4/3,
            '16:9': 16/9,
            '21:9': 21/9,
            '1:1': 1,
            'square': 1,
            'portrait': 3/4,
            'landscape': 4/3
        }

        return standard_ratios.get(aspect_ratio_input.lower())

    def calculate_new_dimensions(self, orig_width, orig_height, scale, aspect_ratio):
        new_width = int(orig_width * scale)

        if aspect_ratio is None:
            new_height = int(orig_height * scale)
        else:
            new_height = int(new_width / aspect_ratio)

        return new_width, new_height

    def remove_background(self, image, background_image, model, alpha_matting, alpha_matting_foreground_threshold,
                      alpha_matting_background_threshold, post_process_mask, chroma_key, chroma_threshold,
                      color_tolerance, background_mode, background_color, output_format="RGBA",
                      invert_mask=False, feather_amount=0, edge_detection=False,
                      edge_thickness=1, edge_color="#FFFFFF", shadow=False, shadow_blur=5,
                      shadow_opacity=0.5, shadow_x_offset=5, shadow_y_offset=5,
                      color_adjustment=False, brightness=1.0, contrast=1.0,
                      saturation=1.0, x_position=0, y_position=0, rotation=0, opacity=1.0,
                      flip_horizontal=False, flip_vertical=False, mask_blur=0, mask_expansion=0,
                      foreground_scale=1.0, foreground_aspect_ratio=None, remove_bg=True,
                      use_custom_dimensions=False, custom_width=None, custom_height=None,
                      output_dimension_source="Foreground", blend_mode="normal"):
        if self.session is None or self.session.model_name != model:
            self.session = new_session(model)

        if not isinstance(background_color, str) or not background_color.startswith('#'):
            background_color = "#000000"

        try:
            bg_color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        except ValueError:
            bg_color = (0, 0, 0, 255)

        try:
            edge_color = tuple(int(edge_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            edge_color = (255, 255, 255)

        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
        original_image = np.array(pil_image)

        if chroma_key != "none":
            chroma_mask = self.apply_chroma_key(original_image, chroma_key, chroma_threshold, color_tolerance)
            input_mask = chroma_mask
        else:
            input_mask = None

        if remove_bg:
            removed_bg = remove(
                pil_image,
                session=self.session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                post_process_mask=post_process_mask,
            )
            rembg_mask = np.array(removed_bg)[:, :, 3]
        else:
            removed_bg = pil_image.convert("RGBA")
            rembg_mask = np.full(pil_image.size[::-1], 255, dtype=np.uint8)

        if input_mask is not None:
            final_mask = cv2.bitwise_and(rembg_mask, input_mask)
        else:
            final_mask = rembg_mask

        final_mask = self.process_mask(final_mask, invert_mask, feather_amount, mask_blur, mask_expansion)

        orig_width, orig_height = pil_image.size
        bg_width, bg_height = background_image.size if background_image else (orig_width, orig_height)

        if use_custom_dimensions and custom_width and custom_height:
            output_width, output_height = int(custom_width), int(custom_height)
        elif output_dimension_source == "Background" and background_image:
            output_width, output_height = bg_width, bg_height
        else:
            output_width, output_height = orig_width, orig_height

        aspect_ratio = self.parse_aspect_ratio(foreground_aspect_ratio)
        new_width, new_height = self.calculate_new_dimensions(orig_width, orig_height, foreground_scale, aspect_ratio)

        fg_image = removed_bg.resize((new_width, new_height), Image.LANCZOS)
        fg_mask = Image.fromarray(final_mask).resize((new_width, new_height), Image.LANCZOS)

        if background_mode == "transparent":
            result = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
        elif background_mode == "color":
            result = Image.new("RGBA", (output_width, output_height), bg_color)
        else:  # background_mode == "image"
            if background_image is not None:
                result = background_image.resize((output_width, output_height), Image.LANCZOS).convert("RGBA")
            else:
                result = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))

        if flip_horizontal:
            fg_image = fg_image.transpose(Image.FLIP_LEFT_RIGHT)
            fg_mask = fg_mask.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_vertical:
            fg_image = fg_image.transpose(Image.FLIP_TOP_BOTTOM)
            fg_mask = fg_mask.transpose(Image.FLIP_TOP_BOTTOM)

        fg_image = fg_image.rotate(rotation, resample=Image.BICUBIC, expand=True)
        fg_mask = fg_mask.rotate(rotation, resample=Image.BICUBIC, expand=True)

        paste_x = x_position + (output_width - fg_image.width) // 2
        paste_y = y_position + (output_height - fg_image.height) // 2

        # Apply blending mode
        if background_mode == "image" and background_image is not None:
            bg_array = np.array(result)
            fg_array = np.array(fg_image)

            # Ensure foreground array matches background dimensions before blending
            if bg_array.shape[:2] != fg_array.shape[:2]:
                # Resize foreground image to match background dimensions
                fg_image = fg_image.resize((output_width, output_height), Image.LANCZOS)
                fg_mask = fg_mask.resize((output_width, output_height), Image.LANCZOS)
                fg_array = np.array(fg_image)

            blended = self.apply_blend_mode(bg_array, fg_array, blend_mode, opacity)
            fg_with_opacity = Image.fromarray(blended)

            # Update paste coordinates since we resized
            paste_x = x_position
            paste_y = y_position
        else:
            fg_rgba = fg_image.convert("RGBA")
            fg_with_opacity = Image.new("RGBA", fg_rgba.size, (0, 0, 0, 0))
            for x in range(fg_rgba.width):
                for y in range(fg_rgba.height):
                    r, g, b, a = fg_rgba.getpixel((x, y))
                    fg_with_opacity.putpixel((x, y), (r, g, b, int(a * opacity)))

        # Ensure mask has same dimensions as image for pasting
        fg_mask_with_opacity = fg_mask.point(lambda p: int(p * opacity))
        if fg_mask_with_opacity.size != fg_with_opacity.size:
            fg_mask_with_opacity = fg_mask_with_opacity.resize(fg_with_opacity.size, Image.LANCZOS)

        if shadow:
            shadow_mask = fg_mask.filter(ImageFilter.GaussianBlur(shadow_blur))
            shadow_image = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
            # 应用阴影偏移
            shadow_x = x_position + shadow_x_offset + (output_width - fg_image.width) // 2
            shadow_y = y_position + shadow_y_offset + (output_height - fg_image.height) // 2
            shadow_image.paste((0, 0, 0, int(255 * shadow_opacity)), (shadow_x, shadow_y), shadow_mask)
            # 先添加阴影，再添加主图像
            result = Image.alpha_composite(result, shadow_image.filter(ImageFilter.GaussianBlur(shadow_blur)))

        # 添加主图像
        result.paste(fg_with_opacity, (paste_x, paste_y), fg_mask_with_opacity)

        if edge_detection:
            edge_mask = cv2.Canny(np.array(fg_mask), 100, 200)
            edge_mask = cv2.dilate(edge_mask, np.ones((edge_thickness, edge_thickness), np.uint8), iterations=1)
            edge_overlay = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
            edge_overlay.paste(Image.new("RGB", fg_image.size, edge_color), (paste_x, paste_y), Image.fromarray(edge_mask))
            result = Image.alpha_composite(result, edge_overlay)

        if color_adjustment:
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(contrast)
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(saturation)

        if output_format == "RGB":
            result = result.convert("RGB")

        return result, fg_mask

    def parse_color(self, color):
        """Safely parse color string to RGB tuple"""
        if isinstance(color, str) and color.startswith('#') and len(color) == 7:
            try:
                return tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
        return (0, 0, 0)  # Default to black if parsing fails

def on_ui():
    with gr.Blocks(analytics_enabled=False) as geeky_remb_tab:
        gr.Markdown("# GeekyRemB: 背景移除与图像处理")

        with gr.Accordion("基本设置", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 前景调整")
                        with gr.Group():
                            blend_mode = gr.Dropdown(
                                label="混合模式",
                                choices=["正常", "正片叠底", "滤色", "叠加", "柔光",
                                        "强光", "差值", "排除", "颜色减淡", "颜色加深"],
                                value="正常"
                            )
                            opacity = gr.Slider(label="不透明度", minimum=0.0, maximum=1.0, value=1.0, step=0.01)

                        foreground_scale = gr.Slider(label="缩放", minimum=0.1, maximum=5.0, value=1.0, step=0.1)
                        foreground_aspect_ratio = gr.Textbox(
                            label="纵横比",
                            placeholder="例如：16:9, 4:3, 1:1, portrait（纵向）, landscape（横向）或留空保持原始比例",
                            value=""
                        )
                        x_position = gr.Slider(label="X 位置", minimum=-1000, maximum=1000, value=0, step=1)
                        y_position = gr.Slider(label="Y 位置", minimum=-1000, maximum=1000, value=0, step=1)
                        rotation = gr.Slider(label="旋转", minimum=-360, maximum=360, value=0, step=0.1)

                        with gr.Row():
                            flip_horizontal = gr.Checkbox(label="水平翻转", value=False)
                            flip_vertical = gr.Checkbox(label="垂直翻转", value=False)

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 背景选项")
                        remove_background = gr.Checkbox(label="移除背景", value=True)
                        background_mode = gr.Radio(label="背景模式", choices=["透明", "纯色", "图片"], value="透明")
                        background_color = gr.ColorPicker(label="背景颜色", value="#000000", visible=False)
                        background_image = gr.Image(label="背景图片", type="pil", visible=False)

        with gr.Accordion("高级设置", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 移除设置")
                    model = gr.Dropdown(label="模型", choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime"], value="u2net")
                    output_format = gr.Radio(label="输出格式", choices=["RGBA", "RGB"], value="RGBA")
                    with gr.Group():
                        alpha_matting = gr.Checkbox(label="启用 Alpha 抠图", value=False)
                        with gr.Group(visible=False) as alpha_matting_group:
                            alpha_matting_foreground_threshold = gr.Slider(label="前景阈值", minimum=0, maximum=255, value=240, step=1)
                            alpha_matting_background_threshold = gr.Slider(label="背景阈值", minimum=0, maximum=255, value=10, step=1)
                            post_process_mask = gr.Checkbox(label="后处理蒙版", value=False)

                with gr.Column():
                    gr.Markdown("### 色键设置")
                    with gr.Group():
                        chroma_key = gr.Dropdown(label="色键", choices=["无", "绿色", "蓝色", "红色"], value="无")
                        with gr.Group(visible=False) as chroma_group:
                            chroma_threshold = gr.Slider(label="色键阈值", minimum=0, maximum=255, value=30, step=1)
                            color_tolerance = gr.Slider(label="颜色容差", minimum=0, maximum=255, value=20, step=1)

                    gr.Markdown("### 蒙版调整")
                    with gr.Group():
                        invert_mask = gr.Checkbox(label="反转蒙版", value=False)
                        feather_amount = gr.Slider(label="羽化程度", minimum=0, maximum=100, value=0, step=1)
                        mask_blur = gr.Slider(label="蒙版模糊", minimum=0, maximum=100, value=0, step=1)
                        mask_expansion = gr.Slider(label="蒙版扩张", minimum=-100, maximum=100, value=0, step=1)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 边缘效果")
                    with gr.Group():
                        edge_detection = gr.Checkbox(label="启用边缘检测", value=False)
                        with gr.Group(visible=False) as edge_group:
                            edge_thickness = gr.Slider(label="边缘粗细", minimum=1, maximum=10, value=1, step=1)
                            edge_color = gr.ColorPicker(label="边缘颜色", value="#FFFFFF", interactive=True)

                with gr.Column():
                    gr.Markdown("### 投影效果")
                    with gr.Group():
                        shadow = gr.Checkbox(label="启用投影", value=False)
                        with gr.Group(visible=False) as shadow_group:
                            shadow_blur = gr.Slider(label="投影模糊", minimum=0, maximum=20, value=5, step=1)
                            shadow_opacity = gr.Slider(label="不透明度", minimum=0.0, maximum=1.0, value=0.5, step=0.1)
                            shadow_x = gr.Slider(label="X 偏移", minimum=-100, maximum=100, value=5, step=1)
                            shadow_y = gr.Slider(label="Y 偏移", minimum=-100, maximum=100, value=5, step=1)

                with gr.Column():
                    gr.Markdown("### 颜色调整")
                    with gr.Group():
                        color_adjustment = gr.Checkbox(label="启用颜色调整", value=False)
                        with gr.Group(visible=False) as color_group:
                            brightness = gr.Slider(label="亮度", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
                            contrast = gr.Slider(label="对比度", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
                            saturation = gr.Slider(label="饱和度", minimum=0.0, maximum=2.0, value=1.0, step=0.1)

            with gr.Row():
                gr.Markdown("### 输出设置")
                use_custom_dimensions = gr.Checkbox(label="使用自定义尺寸", value=False)
                custom_width = gr.Number(label="自定义宽度", value=512, visible=False)
                custom_height = gr.Number(label="自定义高度", value=512, visible=False)
                output_dimension_source = gr.Radio(
                    label="输出尺寸来源",
                    choices=["前景", "背景"],
                    value="前景",
                    visible=True
                )

        # 更新背景模式UI
        def update_background_mode(mode):
            return {
                background_color: gr.update(visible=mode == "纯色"),
                background_image: gr.update(visible=mode == "图片")
            }

        # 更新自定义尺寸UI
        def update_custom_dimensions(use_custom):
            return {
                custom_width: gr.update(visible=use_custom),
                custom_height: gr.update(visible=use_custom),
                output_dimension_source: gr.update(visible=not use_custom)
            }

        # 更新 Alpha 抠图组件状态
        def update_alpha_matting(enabled):
            return {
                alpha_matting_group: gr.update(visible=enabled),
                alpha_matting_foreground_threshold: gr.update(interactive=enabled),
                alpha_matting_background_threshold: gr.update(interactive=enabled),
                post_process_mask: gr.update(interactive=enabled)
            }

        # 更新色键组件状态
        def update_chroma_key(value):
            enabled = value != "无"
            return {
                chroma_group: gr.update(visible=enabled),
                chroma_threshold: gr.update(interactive=enabled),
                color_tolerance: gr.update(interactive=enabled)
            }

        # 更新边缘检测组件状态
        def update_edge_detection(enabled):
            return {
                edge_group: gr.update(visible=enabled),
                edge_thickness: gr.update(interactive=enabled)
            }

        # 更新投影组件状态
        def update_shadow(enabled):
            return {
                shadow_group: gr.update(visible=enabled),
                shadow_blur: gr.update(interactive=enabled),
                shadow_opacity: gr.update(interactive=enabled),
                shadow_x: gr.update(interactive=enabled),
                shadow_y: gr.update(interactive=enabled)
            }

        # 更新颜色调整组件状态
        def update_color_adjustment(enabled):
            return {
                color_group: gr.update(visible=enabled),
                brightness: gr.update(interactive=enabled),
                contrast: gr.update(interactive=enabled),
                saturation: gr.update(interactive=enabled)
            }

        # 绑定UI事件
        background_mode.change(update_background_mode, inputs=[background_mode],
                             outputs=[background_color, background_image])
        use_custom_dimensions.change(update_custom_dimensions, inputs=[use_custom_dimensions],
                                   outputs=[custom_width, custom_height, output_dimension_source])
        alpha_matting.change(update_alpha_matting, inputs=[alpha_matting],
                           outputs=[alpha_matting_group, alpha_matting_foreground_threshold,
                                  alpha_matting_background_threshold, post_process_mask])
        chroma_key.change(update_chroma_key, inputs=[chroma_key],
                         outputs=[chroma_group, chroma_threshold, color_tolerance])
        edge_detection.change(update_edge_detection, inputs=[edge_detection],
                            outputs=[edge_group, edge_thickness])
        shadow.change(update_shadow, inputs=[shadow],
                     outputs=[shadow_group, shadow_blur, shadow_opacity, shadow_x, shadow_y])
        color_adjustment.change(update_color_adjustment, inputs=[color_adjustment],
                              outputs=[color_group, brightness, contrast, saturation])

        return [model, output_format, alpha_matting, alpha_matting_foreground_threshold,
                alpha_matting_background_threshold, post_process_mask, chroma_key,
                chroma_threshold, color_tolerance, background_mode, background_color,
                background_image, invert_mask, feather_amount,
                edge_detection, edge_thickness, edge_color, shadow, shadow_blur,
                shadow_opacity, shadow_x, shadow_y, color_adjustment, brightness, contrast, saturation,
                x_position, y_position, rotation, opacity, flip_horizontal,
                flip_vertical, mask_blur, mask_expansion, foreground_scale,
                foreground_aspect_ratio, remove_background,
                use_custom_dimensions, custom_width, custom_height,
                output_dimension_source, blend_mode]

class Script(scripts.Script):
    def title(self):
        return "GeekyRemB"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        return on_ui()

    def run(self, p, model, output_format, alpha_matting, alpha_matting_foreground_threshold,
            alpha_matting_background_threshold, post_process_mask, chroma_key,
            chroma_threshold, color_tolerance, background_mode, background_color,
            background_image, invert_mask, feather_amount,
            edge_detection, edge_thickness, edge_color, shadow, shadow_blur,
            shadow_opacity, shadow_x, shadow_y, color_adjustment, brightness, contrast, saturation,
            x_position, y_position, rotation, opacity, flip_horizontal,
            flip_vertical, mask_blur, mask_expansion, foreground_scale,
            foreground_aspect_ratio, remove_background,
            use_custom_dimensions, custom_width, custom_height,
            output_dimension_source, blend_mode):

        # 确保颜色值有效
        if not isinstance(background_color, str) or not background_color.startswith('#'):
            background_color = "#000000"
        if not isinstance(edge_color, str) or not edge_color.startswith('#'):
            edge_color = "#FFFFFF"

        # 映射中文选项到原始值
        geeky_remb = GeekyRemB()
        background_mode = geeky_remb.background_mode_map.get(background_mode, "transparent")
        blend_mode = geeky_remb.blend_mode_map.get(blend_mode, "normal")
        chroma_key = geeky_remb.chroma_key_map.get(chroma_key, "none")

        # 生成基本文件名
        basename = f"geeky_rembg_{model}"

        # 创建GeekyRemB实例
        geeky_remb = GeekyRemB()

        args = (model, alpha_matting, alpha_matting_foreground_threshold,
               alpha_matting_background_threshold, post_process_mask, chroma_key,
               chroma_threshold, color_tolerance, background_mode, background_color,
               output_format, invert_mask, feather_amount, edge_detection,
               edge_thickness, edge_color, shadow, shadow_blur, shadow_opacity,
               shadow_x, shadow_y, color_adjustment, brightness, contrast, saturation,
               x_position, y_position, rotation, opacity, flip_horizontal, flip_vertical,
               mask_blur, mask_expansion, foreground_scale, foreground_aspect_ratio,
               remove_background, use_custom_dimensions, custom_width, custom_height,
               output_dimension_source, blend_mode)

        proc = process_images(p)

        for i in range(len(proc.images)):
            # 使用GeekyRemB实例直接处理图像
            result, _ = geeky_remb.remove_background(proc.images[i], background_image, *args)
            proc.images[i] = result

            # 更新处理信息
            try:
                if proc.info is None:
                    proc.info = f"GeekyRemB: {model}"
                elif isinstance(proc.info, str):
                    proc.info += f"\nGeekyRemB: {model}"
                else:
                    proc.info = str(proc.info) + f"\nGeekyRemB: {model}"
            except:
                proc.info = f"GeekyRemB: {model}"

            # 保存图像
            images.save_image(proc.images[i], p.outpath_samples, basename,
                            proc.seed + i, proc.prompt, opts.samples_format,
                            info=proc.info, p=p)

        return proc
