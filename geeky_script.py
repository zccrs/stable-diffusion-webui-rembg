from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageColor
import numpy as np
import cv2
import os
import tempfile

import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
from modules.paths_internal import models_path

class BlendMode:
    @staticmethod
    def normal(target, blend, opacity=1.0):
        return target * (1 - opacity) + blend * opacity

    @staticmethod
    def multiply(target, blend, opacity=1.0):
        result = target * blend
        return target * (1 - opacity) + result * opacity

    @staticmethod
    def screen(target, blend, opacity=1.0):
        result = 1 - (1 - target) * (1 - blend)
        return target * (1 - opacity) + result * opacity

    @staticmethod
    def overlay(target, blend, opacity=1.0):
        result = np.where(target < 0.5, 2 * target * blend, 1 - 2 * (1 - target) * (1 - blend))
        return target * (1 - opacity) + result * opacity

    @staticmethod
    def soft_light(target, blend, opacity=1.0):
        result = np.where(blend < 0.5,
                          target - (1 - 2 * blend) * target * (1 - target),
                          target + (2 * blend - 1) * (np.sqrt(target) - target))
        return target * (1 - opacity) + result * opacity

    @staticmethod
    def hard_light(target, blend, opacity=1.0):
        result = np.where(blend < 0.5,
                          2 * target * blend,
                          1 - 2 * (1 - target) * (1 - blend))
        return target * (1 - opacity) + result * opacity

    @staticmethod
    def difference(target, blend, opacity=1.0):
        result = np.abs(target - blend)
        return target * (1 - opacity) + result * opacity

    @staticmethod
    def exclusion(target, blend, opacity=1.0):
        result = target + blend - 2 * target * blend
        return target * (1 - opacity) + result * opacity

    @staticmethod
    def color_dodge(target, blend, opacity=1.0):
        result = np.where(blend == 1.0, 1.0, np.minimum(1.0, target / (1 - blend)))
        return target * (1 - opacity) + result * opacity

    @staticmethod
    def color_burn(target, blend, opacity=1.0):
        result = np.where(blend == 0.0, 0.0, np.maximum(0.0, 1 - (1 - target) / blend))
        return target * (1 - opacity) + result * opacity

class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.session = None
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
            gr.Markdown("### 背景移除设置")

            models = [
                "None",
                "u2net",
                "u2netp",
                "u2net_human_seg",
                "u2net_cloth_seg",
                "silueta",
                "isnet-general-use",
                "isnet-anime"
            ]

            model = gr.Dropdown(label="模型选择", choices=models, value="u2net")
            return_mask = gr.Checkbox(label="只返回蒙版", value=False)

            with gr.Row():
                background_mode = gr.Radio(label="背景模式", choices=["透明", "颜色", "图像"], value="透明")
                background_color = gr.ColorPicker(label="背景颜色", value="#000000", visible=False)

            background_mode.change(
                fn=lambda mode: gr.update(visible=mode=="颜色"),
                inputs=[background_mode],
                outputs=[background_color]
            )

            # Alpha matting 设置
            alpha_matting = gr.Checkbox(label="Alpha matting", value=False)

            with gr.Row(visible=False) as alpha_mask_row:
                alpha_matting_erode_size = gr.Slider(label="Erode size", minimum=0, maximum=40, step=1, value=10)
                alpha_matting_foreground_threshold = gr.Slider(label="Foreground threshold", minimum=0, maximum=255, step=1, value=240)
                alpha_matting_background_threshold = gr.Slider(label="Background threshold", minimum=0, maximum=255, step=1, value=10)

            alpha_matting.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[alpha_matting],
                outputs=[alpha_mask_row],
            )

            # 效果设置
            with gr.Accordion("高级效果", open=False):
                blend_mode = gr.Dropdown(
                    label="混合模式",
                    choices=list(self.blend_modes.keys()),
                    value="normal"
                )
                opacity = gr.Slider(label="不透明度", minimum=0.0, maximum=1.0, value=1.0, step=0.01)

                with gr.Row():
                    feather_amount = gr.Slider(label="羽化程度", minimum=0, maximum=100, value=0, step=1)
                    mask_blur = gr.Slider(label="蒙版模糊", minimum=0, maximum=100, value=0, step=1)

                with gr.Row():
                    edge_detection = gr.Checkbox(label="边缘检测", value=False)
                    edge_color = gr.ColorPicker(label="边缘颜色", value="#FFFFFF", visible=False)
                    edge_thickness = gr.Slider(label="边缘厚度", minimum=1, maximum=10, value=1, step=1, visible=False)

                edge_detection.change(
                    fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
                    inputs=[edge_detection],
                    outputs=[edge_color, edge_thickness],
                )

                with gr.Row():
                    shadow = gr.Checkbox(label="添加阴影", value=False)
                    shadow_blur = gr.Slider(label="阴影模糊", minimum=0, maximum=20, value=5, step=1, visible=False)
                    shadow_opacity = gr.Slider(label="阴影不透明度", minimum=0.0, maximum=1.0, value=0.5, step=0.1, visible=False)

                shadow.change(
                    fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
                    inputs=[shadow],
                    outputs=[shadow_blur, shadow_opacity],
                )

        overwrite = gr.Checkbox(label="覆盖现有文件", value=False)

        return [model, return_mask, alpha_matting, alpha_matting_foreground_threshold,
                alpha_matting_background_threshold, alpha_matting_erode_size,
                background_mode, background_color, blend_mode, opacity, feather_amount,
                mask_blur, edge_detection, edge_thickness, edge_color, shadow,
                shadow_blur, shadow_opacity, overwrite]

    def parse_color(self, color):
        """安全解析颜色字符串为RGB元组"""
        if isinstance(color, str) and color.startswith('#') and len(color) == 7:
            try:
                return tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
        return (0, 0, 0)  # 解析失败时默认为黑色

    def process_mask(self, mask, feather_amount, mask_blur):
        """处理蒙版的羽化和模糊"""
        if feather_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_amount)

        if mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur)

        return mask

    def apply_blend_mode(self, target, blend, mode="normal", opacity=1.0):
        """应用混合模式"""
        if mode not in self.blend_modes:
            return blend

        target = target.astype(np.float32) / 255
        blend = blend.astype(np.float32) / 255

        result = self.blend_modes[mode](target, blend, opacity)

        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def run(self, p, model, return_mask, alpha_matting, alpha_matting_foreground_threshold,
            alpha_matting_background_threshold, alpha_matting_erode_size, background_mode,
            background_color, blend_mode, opacity, feather_amount, mask_blur, edge_detection,
            edge_thickness, edge_color, shadow, shadow_blur, shadow_opacity, overwrite):

        if not model or model == "None":
            return process_images(p)

        # 设置U2NET模型路径
        if "U2NET_HOME" not in os.environ:
            os.environ["U2NET_HOME"] = os.path.join(models_path, "u2net")

        # 设置文件命名和保存选项
        basename = ""
        if not overwrite:
            basename += f"geeky_rembg_{model}"
            if return_mask:
                basename += "_mask"
        else:
            p.do_not_save_samples = True

        # 创建rembg会话
        session = new_session(model)

        # 处理背景颜色
        if background_mode == "颜色" and background_color:
            bg_color = self.parse_color(background_color) + (255,)  # 添加alpha通道
        else:
            bg_color = (0, 0, 0, 0)  # 默认透明

        # 处理边缘颜色
        if edge_detection and edge_color:
            edge_color_rgb = self.parse_color(edge_color)
        else:
            edge_color_rgb = (255, 255, 255)

        # 正常处理图像
        proc = process_images(p)

        # 对每个处理后的图像应用背景移除
        for i in range(len(proc.images)):
            original_image = proc.images[i]

            # 使用rembg移除背景
            removed_bg = remove(
                original_image,
                session=session,
                only_mask=return_mask,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size
            )

            # 获取蒙版
            if return_mask:
                proc.images[i] = removed_bg
            else:
                # 处理蒙版效果
                mask = np.array(removed_bg)[:, :, 3]
                mask = self.process_mask(mask, feather_amount, mask_blur)

                # 创建结果图像
                if background_mode == "透明":
                    result = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
                elif background_mode == "颜色":
                    result = Image.new("RGBA", original_image.size, bg_color)
                else:  # 背景模式 == "图像"
                    result = original_image.convert("RGBA")

                # 将原始图像转换为RGBA
                fg_image = original_image.convert("RGBA")

                # 创建前景图像
                foreground = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
                fg_array = np.array(fg_image)

                # 应用蒙版到前景
                fg_array[:, :, 3] = mask

                # 如果需要应用混合模式
                if background_mode == "图像" and blend_mode != "normal":
                    bg_array = np.array(result)
                    blended = self.apply_blend_mode(bg_array, fg_array, blend_mode, opacity)
                    result = Image.fromarray(blended)
                else:
                    # 将处理后的前景粘贴到结果
                    foreground = Image.fromarray(fg_array)
                    result.paste(foreground, (0, 0), Image.fromarray(mask))

                # 应用边缘检测
                if edge_detection:
                    edge_mask = cv2.Canny(mask, 100, 200)
                    edge_mask = cv2.dilate(edge_mask, np.ones((edge_thickness, edge_thickness), np.uint8), iterations=1)
                    edge_overlay = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
                    edge_overlay.paste(Image.new("RGB", original_image.size, edge_color_rgb), (0, 0), Image.fromarray(edge_mask))
                    result = Image.alpha_composite(result, edge_overlay)

                # 应用阴影
                if shadow:
                    shadow_mask = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(shadow_blur))
                    shadow_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
                    shadow_image.paste((0, 0, 0, int(255 * shadow_opacity)), (0, 0), shadow_mask)
                    # 在放置前景之前先放置阴影
                    result = Image.alpha_composite(result, shadow_image)

                proc.images[i] = result

            # 处理图像信息元数据
            try:
                if proc.info is None:
                    proc.info = f"GeekyRemB: {model}"
                elif isinstance(proc.info, str):
                    proc.info += f"\nGeekyRemB: {model}"
                elif isinstance(proc.info, dict):
                    info_dict = proc.info.copy()
                    info_dict["GeekyRemB"] = model
                    proc.info = "\n".join([f"{k}: {v}" for k, v in info_dict.items()])
                else:
                    proc.info = str(proc.info) + f"\nGeekyRemB: {model}"
            except Exception as e:
                print(f"无法设置 proc.info: {str(e)}")
                proc.info = f"GeekyRemB: {model}"

            # 保存处理后的图像
            images.save_image(proc.images[i], p.outpath_samples, basename,
                proc.seed + i, proc.prompt, opts.samples_format, info=proc.info, p=p)

        return proc
