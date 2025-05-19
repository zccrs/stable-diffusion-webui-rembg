import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
from modules.paths_internal import models_path
import rembg

class Script(scripts.Script):
    def title(self):
        return "Remove Background"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        models = [
            "None",
            "isnet-general-use",
            "u2net",
            "u2netp",
            "u2net_human_seg",
            "u2net_cloth_seg",
            "silueta",
            "isnet-anime",
        ]

        model = gr.Dropdown(label="Remove background model", choices=models, value="u2net")
        return_mask = gr.Checkbox(label="Return mask", value=False)
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

        overwrite = gr.Checkbox(label="Overwrite existing files", value=False)
        return [model, return_mask, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size, overwrite]

    def run(self, p, model, return_mask, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size, overwrite):
        if not model or model == "None":
            return process_images(p)

        if "U2NET_HOME" not in os.environ:
            os.environ["U2NET_HOME"] = os.path.join(models_path, "u2net")

        basename = ""
        if not overwrite:
            basename += f"rembg_{model}"
            if return_mask:
                basename += "_mask"
            if alpha_matting:
                basename += "_alpha"
        else:
            p.do_not_save_samples = True

        proc = process_images(p)

        for i in range(len(proc.images)):
            proc.images[i] = rembg.remove(
                proc.images[i],
                session=rembg.new_session(model),
                only_mask=return_mask,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
            )

            # 处理 proc.info 信息
            try:
                # 确保 proc.info 是字符串类型，WebUI 期望处理字符串
                if proc.info is None:
                    proc.info = f"Rembg: {model}"
                elif isinstance(proc.info, str):
                    proc.info += f"\nRembg: {model}"
                elif isinstance(proc.info, dict):
                    # 如果是字典，转换为字符串
                    info_dict = proc.info.copy()
                    info_dict["Rembg"] = model
                    proc.info = "\n".join([f"{k}: {v}" for k, v in info_dict.items()])
                else:
                    # 其他类型，直接转为字符串
                    proc.info = str(proc.info) + f"\nRembg: {model}"
            except Exception as e:
                print(f"无法设置 proc.info: {str(e)}")
                # 转换失败时确保是字符串
                proc.info = f"Rembg: {model}"

            images.save_image(proc.images[i], p.outpath_samples, basename,
                proc.seed + i, proc.prompt, opts.samples_format, info=proc.info, p=p)

        return proc
