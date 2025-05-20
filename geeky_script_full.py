import modules.scripts as scripts
import gradio as gr
import os
import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageColor
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import queue
from threading import Thread
import torch
import tempfile

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
from modules.paths_internal import models_path

# BlendMode class with alpha channel handling
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
            return np.where(t > 0.5, 1 - 2 * (1 - t) * (1 - b), 2 * t * b)
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

class Script(scripts.Script):

    def __init__(self):
        super().__init__()
        self.session = None
        if "U2NET_HOME" not in os.environ:
            os.environ["U2NET_HOME"] = os.path.join(models_path, "u2net")
        self.processing = False
        self.use_gpu = torch.cuda.is_available()
        self.frame_cache = {}
        self.max_cache_size = 100
        self.batch_size = 4  # 基于内存限制调整
        self.max_workers = 4  # 基于CPU核心数调整
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
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

    def title(self):
        return "GeekyRemB 高级背景移除"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 移除设置")
                    models = [
                        "None",
                        "u2net",
                        "u2netp",
                        "u2net_human_seg",
                        "u2net_cloth_seg",
                        "silueta",
                        "isnet-general-use",
                        "isnet-anime",
                    ]
                    model = gr.Dropdown(label="背景移除模型", choices=models, value="u2net")
                    return_mask = gr.Checkbox(label="只返回蒙版", value=False)

                    with gr.Row():
                        background_mode = gr.Radio(label="背景模式", choices=["透明", "颜色", "图像", "视频"], value="透明")
                        background_color = gr.ColorPicker(label="背景颜色", value="#000000", visible=False)

                    background_mode.change(
                        fn=lambda mode: gr.update(visible=mode=="颜色"),
                        inputs=[background_mode],
                        outputs=[background_color]
                    )

                with gr.Column():
                    gr.Markdown("### Alpha Matting 设置")
                    alpha_matting = gr.Checkbox(label="使用 Alpha Matting", value=False)
                    with gr.Group(visible=False) as alpha_settings:
                        alpha_matting_foreground_threshold = gr.Slider(label="前景阈值", minimum=0, maximum=255, value=240, step=1)
                        alpha_matting_background_threshold = gr.Slider(label="背景阈值", minimum=0, maximum=255, value=10, step=1)
                        alpha_matting_erode_size = gr.Slider(label="腐蚀大小", minimum=0, maximum=40, value=10, step=1)

                    alpha_matting.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[alpha_matting],
                        outputs=[alpha_settings],
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Chroma Key 设置")
                    chroma_key = gr.Dropdown(label="色度键控", choices=["none", "green", "blue", "red"], value="none")
                    with gr.Group() as chroma_settings:
                        chroma_threshold = gr.Slider(label="阈值", minimum=0, maximum=255, value=30, step=1)
                        color_tolerance = gr.Slider(label="颜色容差", minimum=0, maximum=255, value=20, step=1)

                    chroma_key.change(
                        fn=lambda x: gr.update(visible=x!="none"),
                        inputs=[chroma_key],
                        outputs=[chroma_settings],
                    )

                with gr.Column():
                    gr.Markdown("### 前景尺寸调整")
                    foreground_scale = gr.Slider(label="缩放", minimum=0.1, maximum=5.0, value=1.0, step=0.1)
                    foreground_aspect_ratio = gr.Textbox(
                        label="纵横比",
                        placeholder="例如: 16:9, 4:3, 1:1, portrait, landscape, 或留空保持原始比例",
                        value=""
                    )
                    use_custom_dimensions = gr.Checkbox(label="使用自定义尺寸", value=False)
                    with gr.Group(visible=False) as dimension_settings:
                        custom_width = gr.Number(label="自定义宽度", value=512)
                        custom_height = gr.Number(label="自定义高度", value=512)

                    use_custom_dimensions.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[use_custom_dimensions],
                        outputs=[dimension_settings],
                    )

            with gr.Accordion("高级设置", open=False):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 混合设置")
                        blend_mode = gr.Dropdown(
                            label="混合模式",
                            choices=list(self.blend_modes.keys()),
                            value="normal"
                        )
                        opacity = gr.Slider(label="不透明度", minimum=0.0, maximum=1.0, value=1.0, step=0.01)

                    with gr.Column():
                        gr.Markdown("### 特效设置")
                        edge_detection = gr.Checkbox(label="边缘检测", value=False)
                        with gr.Group(visible=False) as edge_settings:
                            edge_thickness = gr.Slider(label="边缘厚度", minimum=1, maximum=10, value=1, step=1)
                            edge_color = gr.ColorPicker(label="边缘颜色", value="#FFFFFF")

                        shadow = gr.Checkbox(label="阴影", value=False)
                        with gr.Group(visible=False) as shadow_settings:
                            shadow_blur = gr.Slider(label="阴影模糊", minimum=0, maximum=20, value=5, step=1)
                            shadow_opacity = gr.Slider(label="阴影不透明度", minimum=0.0, maximum=1.0, value=0.5, step=0.1)

                        color_adjustment = gr.Checkbox(label="颜色调整", value=False)
                        with gr.Group(visible=False) as color_settings:
                            brightness = gr.Slider(label="亮度", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
                            contrast = gr.Slider(label="对比度", minimum=0.0, maximum=2.0, value=1.0, step=0.1)
                            saturation = gr.Slider(label="饱和度", minimum=0.0, maximum=2.0, value=1.0, step=0.1)

                        edge_detection.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[edge_detection],
                            outputs=[edge_settings],
                        )

                        shadow.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[shadow],
                            outputs=[shadow_settings],
                        )

                        color_adjustment.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[color_adjustment],
                            outputs=[color_settings],
                        )

                    with gr.Column():
                        gr.Markdown("### 变换设置")
                        x_position = gr.Slider(label="X 位置", minimum=-1000, maximum=1000, value=0, step=1)
                        y_position = gr.Slider(label="Y 位置", minimum=-1000, maximum=1000, value=0, step=1)
                        rotation = gr.Slider(label="旋转", minimum=-360, maximum=360, value=0, step=0.1)

                        with gr.Row():
                            flip_horizontal = gr.Checkbox(label="水平翻转", value=False)
                            flip_vertical = gr.Checkbox(label="垂直翻转", value=False)

                    with gr.Column():
                        gr.Markdown("### 蒙版设置")
                        feather_amount = gr.Slider(label="羽化程度", minimum=0, maximum=100, value=0, step=1)
                        mask_blur = gr.Slider(label="蒙版模糊", minimum=0, maximum=100, value=0, step=1)
                        mask_expansion = gr.Slider(label="蒙版扩张", minimum=-100, maximum=100, value=0, step=1)
                        invert_mask = gr.Checkbox(label="反转蒙版", value=False)

        with gr.Row():
            gr.Markdown("### 输出设置")
            output_format = gr.Radio(label="输出格式", choices=["RGBA", "RGB"], value="RGBA")
            output_dimension_source = gr.Radio(
                label="输出尺寸来源",
                choices=["前景", "背景"],
                value="前景",
            )
            overwrite = gr.Checkbox(label="覆盖现有文件", value=False)

        return [model, return_mask, alpha_matting, alpha_matting_foreground_threshold,
                alpha_matting_background_threshold, alpha_matting_erode_size, chroma_key,
                chroma_threshold, color_tolerance, background_mode, background_color,
                output_format, invert_mask, feather_amount, edge_detection,
                edge_thickness, edge_color, shadow, shadow_blur, shadow_opacity,
                color_adjustment, brightness, contrast, saturation, x_position,
                y_position, rotation, opacity, flip_horizontal, flip_vertical,
                mask_blur, mask_expansion, foreground_scale, foreground_aspect_ratio,
                use_custom_dimensions, custom_width, custom_height,
                output_dimension_source, blend_mode, overwrite]

    # 核心背景处理功能
    def remove_background(self, image, model, return_mask=False, alpha_matting=False,
                        alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10,
                        alpha_matting_erode_size=10, background_mode="transparent",
                        background_color="#000000", blend_mode="normal", opacity=1.0,
                        feather_amount=0, mask_blur=0, edge_detection=False,
                        edge_thickness=1, edge_color="#FFFFFF", shadow=False,
                        shadow_blur=5, shadow_opacity=0.5, mask_expansion=0):

        if self.session is None or self.session.model_name != model:
            self.session = new_session(model)

        # 处理颜色设置
        try:
            bg_color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
        except:
            bg_color = (0, 0, 0, 255)

        try:
            edge_color_rgb = tuple(int(edge_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        except:
            edge_color_rgb = (255, 255, 255)

        # 转换为PIL图像
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # 移除背景
        removed = remove(
            image,
            session=self.session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )

        if return_mask:
            return removed

        # 提取蒙版
        mask = np.array(removed)[:, :, 3]

        # 处理蒙版
        if mask_expansion != 0:
            kernel = np.ones((abs(mask_expansion), abs(mask_expansion)), np.uint8)
            if mask_expansion > 0:
                mask = cv2.dilate(mask, kernel)
            else:
                mask = cv2.erode(mask, kernel)

        if feather_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_amount)

        if mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur)

        # 创建结果图像
        if background_mode == "透明":
            result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        elif background_mode == "颜色":
            result = Image.new("RGBA", image.size, bg_color)
        else:  # 图像模式
            result = image.convert("RGBA")

        # 应用混合模式
        fg_rgba = removed.convert("RGBA")
        bg_array = np.array(result)
        fg_array = np.array(fg_rgba)

        if blend_mode != "normal":
            blended = self.blend_modes[blend_mode](bg_array, fg_array, opacity)
            result = Image.fromarray(blended)
        else:
            # 使用透明度混合
            fg_array[..., 3] = (fg_array[..., 3] * opacity).astype(np.uint8)
            fg_with_opacity = Image.fromarray(fg_array)
            result.paste(fg_with_opacity, (0, 0), fg_with_opacity)

        # 应用边缘检测
        if edge_detection:
            edge_mask = cv2.Canny(mask, 100, 200)
            edge_mask = cv2.dilate(edge_mask, np.ones((edge_thickness, edge_thickness), np.uint8))
            edge_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            edge_overlay.paste(Image.new("RGB", image.size, edge_color_rgb), mask=Image.fromarray(edge_mask))
            result = Image.alpha_composite(result, edge_overlay)

        # 应用阴影
        if shadow:
            shadow_mask = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(shadow_blur))
            shadow_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
            shadow_image.paste((0, 0, 0, int(255 * shadow_opacity)), mask=shadow_mask)
            shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(shadow_blur))
            # 在结果图像下方添加阴影
            shadow_base = Image.new("RGBA", image.size, (0, 0, 0, 0))
            shadow_base = Image.alpha_composite(shadow_base, shadow_image)
            shadow_base = Image.alpha_composite(shadow_base, result)
            result = shadow_base

        return result

    def run(self, p, model, return_mask, alpha_matting, alpha_matting_foreground_threshold,
            alpha_matting_background_threshold, alpha_matting_erode_size, chroma_key,
            chroma_threshold, color_tolerance, background_mode, background_color,
            output_format, invert_mask, feather_amount, edge_detection,
            edge_thickness, edge_color, shadow, shadow_blur, shadow_opacity,
            color_adjustment, brightness, contrast, saturation, x_position,
            y_position, rotation, opacity, flip_horizontal, flip_vertical,
            mask_blur, mask_expansion, foreground_scale, foreground_aspect_ratio,
            use_custom_dimensions, custom_width, custom_height,
            output_dimension_source, blend_mode, overwrite=False):  # 添加overwrite参数

        if not model or model == "None":
            return process_images(p)

        # 设置U2NET模型路径
        if "U2NET_HOME" not in os.environ:
            os.environ["U2NET_HOME"] = os.path.join(models_path, "u2net")

        basename = ""
        if not overwrite:
            basename += f"geeky_rembg_{model}"
            if return_mask:
                basename += "_mask"
        else:
            p.do_not_save_samples = True

        proc = process_images(p)

        for i in range(len(proc.images)):
            # 获取原始图像
            original_image = proc.images[i]

            # 首先应用ChromaKey(如果启用)
            if chroma_key != "none":
                chroma_mask = self.apply_chroma_key(original_image, chroma_key, chroma_threshold, color_tolerance)
                input_mask = chroma_mask
            else:
                input_mask = None

            # 执行背景移除
            removed = remove(
                original_image,
                session=self.session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
            )

            # 如果只需要返回蒙版
            if return_mask:
                proc.images[i] = removed
                continue

            # 提取并处理蒙版
            mask = np.array(removed)[:, :, 3]

            # 合并ChromaKey蒙版(如果有)
            if input_mask is not None:
                mask = cv2.bitwise_and(mask, input_mask)

            # 应用高级蒙版处理
            mask = self.process_mask(mask, invert_mask, feather_amount, mask_blur, mask_expansion)

            # 获取原始尺寸
            orig_width, orig_height = original_image.size

            # 计算新尺寸
            if use_custom_dimensions and custom_width and custom_height:
                output_width, output_height = int(custom_width), int(custom_height)
            else:
                output_width, output_height = orig_width, orig_height

            # 应用前景变换(缩放、纵横比)
            aspect_ratio = self.parse_aspect_ratio(foreground_aspect_ratio)
            new_width, new_height = self.calculate_new_dimensions(orig_width, orig_height, foreground_scale, aspect_ratio)

            fg_image = original_image.resize((new_width, new_height), Image.LANCZOS)
            fg_mask = Image.fromarray(mask).resize((new_width, new_height), Image.LANCZOS)

            # 创建结果图像
            if background_mode == "透明":
                result = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
            elif background_mode == "颜色":
                try:
                    bg_color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
                except:
                    bg_color = (0, 0, 0, 255)
                result = Image.new("RGBA", (output_width, output_height), bg_color)
            else:  # background_mode == "图像" 或 "视频"
                result = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))

            # 应用翻转
            if flip_horizontal:
                fg_image = fg_image.transpose(Image.FLIP_LEFT_RIGHT)
                fg_mask = fg_mask.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_vertical:
                fg_image = fg_image.transpose(Image.FLIP_TOP_BOTTOM)
                fg_mask = fg_mask.transpose(Image.FLIP_TOP_BOTTOM)

            # 应用旋转
            fg_image = fg_image.rotate(rotation, resample=Image.BICUBIC, expand=True)
            fg_mask = fg_mask.rotate(rotation, resample=Image.BICUBIC, expand=True)

            # 计算粘贴位置
            paste_x = x_position + (output_width - fg_image.width) // 2
            paste_y = y_position + (output_height - fg_image.height) // 2

            # 应用混合模式和不透明度
            fg_rgba = fg_image.convert("RGBA")
            fg_array = np.array(fg_rgba)

            if blend_mode != "normal":
                bg_array = np.array(result)
                blended = self.apply_blend_mode(bg_array, fg_array, blend_mode, opacity)
                fg_with_opacity = Image.fromarray(blended)
            else:
                fg_array[..., 3] = (fg_array[..., 3] * opacity).astype(np.uint8)
                fg_with_opacity = Image.fromarray(fg_array)

            # 应用Alpha混合
            result.paste(fg_with_opacity, (paste_x, paste_y), fg_mask)

            # 应用边缘检测
            if edge_detection:
                try:
                    edge_color_rgb = tuple(int(edge_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                except:
                    edge_color_rgb = (255, 255, 255)

                edge_mask = cv2.Canny(np.array(fg_mask), 100, 200)
                edge_mask = cv2.dilate(edge_mask, np.ones((edge_thickness, edge_thickness), np.uint8))
                edge_overlay = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
                edge_overlay.paste(Image.new("RGB", fg_image.size, edge_color_rgb), (paste_x, paste_y), Image.fromarray(edge_mask))
                result = Image.alpha_composite(result, edge_overlay)

            # 应用阴影
            if shadow:
                shadow_mask = fg_mask.filter(ImageFilter.GaussianBlur(shadow_blur))
                shadow_image = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
                shadow_image.paste((0, 0, 0, int(255 * shadow_opacity)), (paste_x, paste_y), shadow_mask)
                result = Image.alpha_composite(result, shadow_image.filter(ImageFilter.GaussianBlur(shadow_blur)))

            # 应用颜色调整
            if color_adjustment:
                enhancer = ImageEnhance.Brightness(result)
                result = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(result)
                result = enhancer.enhance(contrast)
                enhancer = ImageEnhance.Color(result)
                result = enhancer.enhance(saturation)

            # 转换输出格式
            if output_format == "RGB":
                result = result.convert("RGB")

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
                proc.seed + i, proc.prompt, opts.samples_format, info=proc.info, p=p)

        return proc

    def process_frame_batch(self, frames, background_frames, *args):
        """处理多个帧"""
        futures = []
        for frame, bg_frame in zip(frames, background_frames):
            future = self.executor.submit(self.process_frame, frame, bg_frame, *args)
            futures.append(future)
        return [future.result() for future in futures]

    def process_frame(self, frame, background_frame=None, *args):
        """处理单个视频帧"""
        if isinstance(frame, np.ndarray):
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            pil_frame = frame

        args = list(args)

        if len(args) > 9:  # 处理背景颜色
            bg_color = self.parse_color(args[9])
            args[9] = f"#{bg_color[0]:02x}{bg_color[1]:02x}{bg_color[2]:02x}"

        if len(args) > 14:  # 处理边缘颜色
            edge_color = self.parse_color(args[14])
            args[14] = f"#{edge_color[0]:02x}{edge_color[1]:02x}{edge_color[2]:02x}"

        if background_frame is not None:
            if isinstance(background_frame, np.ndarray):
                background_frame = Image.fromarray(cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB))

        args = tuple(args)
        processed_frame, _ = self.remove_background(pil_frame, background_frame, *args)
        return cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)

    def process_video(self, input_path, output_path, background_video_path, *args):
        """处理视频"""
        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            bg_cap = None
            if background_video_path:
                bg_cap = cv2.VideoCapture(background_video_path)
                bg_total_frames = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_queue = queue.Queue(maxsize=self.batch_size * 2)
            result_queue = queue.Queue()

            # 如果支持则使用GPU加速编码
            if self.use_gpu:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            def read_frames():
                frame_idx = 0
                while frame_idx < total_frames:
                    frames = []
                    bg_frames = []
                    for _ in range(self.batch_size):
                        if frame_idx >= total_frames:
                            break
                        ret, frame = cap.read()
                        if not ret:
                            break

                        bg_frame = None
                        if bg_cap is not None:
                            bg_frame_idx = frame_idx % bg_total_frames
                            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_idx)
                            bg_ret, bg_frame = bg_cap.read()
                            if bg_ret:
                                bg_frame = cv2.resize(bg_frame, (width, height))

                        frames.append(frame)
                        bg_frames.append(bg_frame)
                        frame_idx += 1

                    if frames:
                        frame_queue.put((frames, bg_frames))
                frame_queue.put(None)

            def process_frames():
                while True:
                    batch = frame_queue.get()
                    if batch is None:
                        result_queue.put(None)
                        break
                    frames, bg_frames = batch
                    processed_frames = self.process_frame_batch(frames, bg_frames, *args)
                    result_queue.put(processed_frames)

            read_thread = Thread(target=read_frames)
            process_thread = Thread(target=process_frames)
            read_thread.start()
            process_thread.start()

            with tqdm(total=total_frames, desc="正在处理视频") as pbar:
                while True:
                    processed_batch = result_queue.get()
                    if processed_batch is None:
                        break
                    for processed_frame in processed_batch:
                        out.write(processed_frame)
                        pbar.update(1)

            read_thread.join()
            process_thread.join()
            cap.release()
            if bg_cap:
                bg_cap.release()
            out.release()

            # 优化最终视频编码
            temp_output = output_path + "_temp.mp4"
            os.rename(output_path, temp_output)
            if self.use_gpu:
                os.system(f'ffmpeg -y -i "{temp_output}" -c:v h264_nvenc -preset p7 -tune hq -crf 23 "{output_path}"')
            else:
                os.system(f'ffmpeg -y -i "{temp_output}" -c:v libx264 -preset faster -crf 23 "{output_path}"')
            if os.path.exists(temp_output):
                os.remove(temp_output)

        except Exception as e:
            print(f"处理视频时出错: {str(e)}")
            raise

        finally:
            if 'cap' in locals():
                cap.release()
            if 'bg_cap' in locals():
                bg_cap.release()
            if 'out' in locals():
                out.release()

    def apply_chroma_key(self, image, color, threshold, color_tolerance=20):
        """应用色度键控(绿幕/蓝幕)效果"""
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
        """更高级的蒙版处理"""
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
        """解析纵横比设置"""
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
        """计算新的尺寸"""
        new_width = int(orig_width * scale)

        if aspect_ratio is None:
            new_height = int(orig_height * scale)
        else:
            new_height = int(new_width / aspect_ratio)

        return new_width, new_height
