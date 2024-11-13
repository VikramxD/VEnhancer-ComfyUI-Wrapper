import argparse
import os
import random

import gradio as gr

from enhance_a_video import VEnhancer

examples = [
    ["prompts/astronaut.mp4", "An astronaut flying in space, featuring a steady and smooth perspective", 4, 24, 250],
    ["prompts/cat.mp4", "A cat wearing sunglasses at a pool", 4, 24, 250],
    ["prompts/fish.mp4", "Clown fish swimming through the coral reef", 4, 24, 300],
    ["prompts/iron_man.mp4", "Iron Man flying in the sky", 4, 24, 250],
    ["prompts/raccoon.mp4", "A cute raccoon playing guitar in a boat on the ocean", 4, 24, 250],
    ["prompts/shanghai.mp4", "The bund Shanghai by Hokusai, in the style of Ukiyo", 4, 24, 250],
    ["prompts/gwen.mp4", "Gwen Stacy reading a book, black and white", 4, 24, 250],
]


def venhancer_demo(result_dir="./tmp/", version="v1"):
    css = """#input_video {max-width: 1024px !important} #output_vid {max-width: 2048px; max-height:1280px}"""
    venhancer = VEnhancer(result_dir, version=version)
    with gr.Blocks(analytics_enabled=False, css=css) as venhancer_iface:
        gr.Markdown(
            "<div align='center'> <h1> VEnhancer </span> </h1> \
                     <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2407.07667'> [ArXiv] </a>\
                     <a style='font-size:18px;color: #000000' href='https://vchitect.github.io/VEnhancer-project/'> [Project Page] </a> \
                     <a style='font-size:18px;color: #000000' href='https://github.com/Vchitect/VEnhancer'> [Github] </a> </div>"
        )
        with gr.Tab(label="VEnhancer"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            input_video = gr.Video(label="Input Video", elem_id="input_video")
                        with gr.Row():
                            input_text = gr.Text(label="Prompts")
                        with gr.Row():
                            up_scale = gr.Slider(
                                minimum=1.0, maximum=8.0, step=0.1, label="up scale", value=4, elem_id="up_scale"
                            )
                        with gr.Row():
                            target_fps = gr.Slider(
                                minimum=8, maximum=60, step=1, elem_id="target_fps", label="target fps", value=24
                            )
                        with gr.Row():
                            noise_aug = gr.Slider(
                                minimum=50, maximum=300, step=1, elem_id="noise_aug", label="noise aug", value=250
                            )
                        end_btn = gr.Button("Generate")
                    with gr.Row():
                        output_video = gr.Video(
                            label="Generated Video", elem_id="output_vid", autoplay=True, show_share_button=True
                        )

                gr.Examples(
                    examples=examples,
                    inputs=[input_video, input_text, up_scale, target_fps, noise_aug],
                    outputs=[output_video],
                    fn=venhancer.enhance_a_video,
                    cache_examples=False,
                )
            end_btn.click(
                inputs=[input_video, input_text, up_scale, target_fps, noise_aug],
                outputs=[output_video],
                fn=venhancer.enhance_a_video,
            )

    return venhancer_iface


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", type=str, default="v1", choices=["v1", "v2"], help="select the checkpoint version."
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    result_dir = os.path.join("./", "results")
    venhancer_iface = venhancer_demo(result_dir, args.version)
    venhancer_iface.queue(max_size=12)
    venhancer_iface.launch(max_threads=1)
