#!/usr/bin/env python
"""Demo app for https://github.com/ziqihuangg/ReVersion.
The code in this repo is partly adapted from the following repository:
https://github.com/ziqihuangg/ReVersion
S-Lab License 1.0
Copyright 2023 S-Lab
Redistribution and use for non-commercial purpose in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
4. In the event that redistribution and/or use for commercial purpose in source or binary forms, with or without modification is required, please contact the contributor(s) of the work.
"""

from __future__ import annotations
import sys
import os
import pathlib

import argparse
import gradio as gr
import torch

from inference import inference_fn


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', type=str, default='cpu')
#     parser.add_argument('--theme', type=str)
#     parser.add_argument('--share', action='store_true')
#     parser.add_argument('--port', type=int)
#     parser.add_argument('--disable-queue',
#                         dest='enable_queue',
#                         action='store_false')
#     return parser.parse_args()


TITLE = '# ReVersion'
DESCRIPTION = '''
This is a demo for **ReVersion: Diffusion-Based Relation Inversion from Images**
[[Paper](https://arxiv.org/abs/2303.13495)] | [[Project Page](https://ziqihuangg.github.io/projects/reversion.html)] | [[GitHub Code](https://github.com/ziqihuangg/ReVersion)] | [[Video](https://www.youtube.com/watch?v=pkal3yjyyKQ)]
It is recommended to upgrade to GPU in Settings after duplicating this space to use it. <a href="https://huggingface.co/spaces/Ziqi/ReVersion?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
'''
DETAILDESCRIPTION='''
ReVersion
'''
DETAILDESCRIPTION='''
We propose a new task, **Relation Inversion**: Given a few exemplar images, where a relation co-exists in every image, we aim to find a relation prompt **\<R>** to capture this interaction, and apply the relation to new entities to synthesize new scenes.
Here we give several pre-trained relation prompts for you to play with. You can choose a set of exemplar images from the examples, and use **\<R>** in your prompt for relation-specific text-to-image generation.
'''


ORIGINAL_SPACE_ID = 'Ziqi/ReVersion'
SPACE_ID = os.getenv('SPACE_ID', ORIGINAL_SPACE_ID)


if os.getenv('SYSTEM') == 'spaces' and SPACE_ID != ORIGINAL_SPACE_ID:
    SETTINGS = f'<a href="https://huggingface.co/spaces/{SPACE_ID}/settings">Settings</a>'

else:
    SETTINGS = 'Settings'
CUDA_NOT_AVAILABLE_WARNING = f'''# Attention - Running on CPU.
<center>
You can assign a GPU in the {SETTINGS} tab if you are running this on HF Spaces.
"T4 small" is sufficient to run this demo.
</center>
'''

# os.system("git clone https://github.com/ziqihuangg/ReVersion")
# sys.path.append("ReVersion")

def show_warning(warning_text: str) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown(warning_text)
    return demo

def set_example_image(example: list):
    return gr.update(value=example[0])

def create_inference_demo(func: inference_fn) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                exemplar_img = gr.Image(
                    label='Exemplar Images',
                    type='pil',
                    interaction=False
                )
                # paths = sorted(pathlib.Path('exemplars').glob('*.jpg'))
                # exemplar_dataset = gr.Dataset(components=[exemplar_img],
                #                             samples=[[path.as_posix()]
                #                                      for path in paths])
                exemplar_dataset = gr.Dataset(
                    components=[exemplar_img],
                    samples = [
                        ['exemplars/painted_on.jpg'],
                        ['exemplars/carved_by.jpg'],
                        ['exemplars/inside.jpg']
                    ]
                )

                # model_id = gr.Dropdown(
                #     choices=['painted_on', 'carved_by', 'inside'],
                #     value='painted_on',
                #     label='Relation',
                #     visible=True)
                prompt = gr.Textbox(
                    label='Prompt',
                    max_lines=1,
                    placeholder='Example: "cat <R> stone"')
                # placeholder_string = gr.Textbox(
                #     label='Placeholder String',
                #     max_lines=1,
                #     placeholder='Example: "<R>"')

                with gr.Accordion('Other Parameters', open=False):
                    num_samples = gr.Slider(label='Number of Images to Generate',
                                               minimum=4,
                                               maximum=8,
                                               step=2,
                                               value=6)
                    guidance_scale = gr.Slider(label='Classifier-Free Guidance Scale',
                                               minimum=0,
                                               maximum=50,
                                               step=0.1,
                                               value=7.5)
                    ddim_steps = gr.Slider(label='Number of DDIM Sampling Steps',
                                               minimum=10,
                                               maximum=100,
                                               step=1,
                                               value=50)
                run_button = gr.Button('Generate')

            with gr.Column():
                result = gr.Image(label='Result')


        exemplar_dataset.click(fn=set_example_image,
                                inputs=exemplar_dataset,
                                outputs=exemplar_dataset.components,
                                queue=False)
        prompt.submit(
            fn=func,
            # inputs=[
            #     model_id,
            #     prompt,
            #     num_samples,
            #     guidance_scale,
            #     ddim_steps
            # ],
            inputs=[
                exemplar_dataset,
                prompt,
                num_samples,
                guidance_scale,
                ddim_steps
            ],
            outputs=result,
            queue=False
        )

        run_button.click(
            fn=func,
            # inputs=[
            #     model_id,
            #     prompt,
            #     num_samples,
            #     guidance_scale,
            #     ddim_steps
            # ],
            inputs=[
                exemplar_dataset,
                prompt,
                num_samples,
                guidance_scale,
                ddim_steps
            ],
            outputs=result,
            queue=False
        )
    return demo


# args = parse_args()
# args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('*** Now using %s.'%(args.device))
if torch.cuda.is_available():
    print('*** Now using %s.'%('cuda'))
else:
    print('*** Now using %s.'%('cpu'))

with gr.Blocks(css='style.css') as demo:
    if not torch.cuda.is_available():
        show_warning(CUDA_NOT_AVAILABLE_WARNING)

    gr.Markdown(TITLE)
    gr.Markdown(DESCRIPTION)
    gr.Markdown(DETAILDESCRIPTION)

    with gr.Tabs():
        with gr.TabItem('Relation-Specific Text-to-Image Generation'):
            create_inference_demo(inference_fn)

demo.queue(default_enabled=False).launch(share=False)

# demo.launch(
#     enable_queue=args.enable_queue,
#     server_port=args.port,
#     share=args.share
# )
# demo.queue(default_enabled=False).launch(server_port=args.port, share=args.share)
