import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import gradio as gr
from PIL import Image
import torch
from torch import nn
import warnings
from diffusers import StableDiffusionXLPipeline
from src.utils.device_utils import device
from src.utils.config_utils import load_config
import tempfile
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import io
import cv2

warnings.filterwarnings("ignore", category=RuntimeWarning, module="diffusers.image_processor")

config = load_config()
MODEL_PATH = os.path.join(config['training']['checkpoint_dir'], "checkpoint-best")

class TilingConv2d(nn.Conv2d):
    def forward(self, x):
        batch, channel, height, width = x.shape
        if self.kernel_size[0] > 1:
            x = torch.cat([x[:, :, -self.padding[0]:], x, x[:, :, :self.padding[0]]], dim=2)
        if self.kernel_size[1] > 1:
            x = torch.cat([x[:, :, :, -self.padding[1]:], x, x[:, :, :, :self.padding[1]]], dim=3)
        return nn.functional.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)

def make_unet_tiling_compatible(unet):
    for module in unet.modules():
        if isinstance(module, nn.Conv2d):
            module.__class__ = TilingConv2d
    return unet

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16,
    use_safetensors=True,
).to(device)

original_unet = pipe.unet

def hex_to_rgb(hex_color):
    return np.array([int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)])

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

@torch.no_grad()
def generate_pattern(prompt, negative_prompt, num_images=1, guidance_scale=7.5, num_inference_steps=50, width=1024, height=1024, enable_tiling=False):
    if enable_tiling:
        tiling_prompt = "seamless pattern, tileable, repeating pattern, "
        prompt = tiling_prompt + prompt
        tiling_negative = "border, frame, text, watermark, signature, "
        negative_prompt = tiling_negative + negative_prompt
        pipe.unet = make_unet_tiling_compatible(original_unet)
    else:
        pipe.unet = original_unet

    width = (width // 64) * 64
    height = (height // 64) * 64

    images = pipe(
        prompt=[prompt] * num_images,
        negative_prompt=[negative_prompt] * num_images,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
    ).images
    
    return images

def limit_colors(image, n_colors, user_colors=None):
    img_array = np.array(image)
    h, w, d = img_array.shape
    reshaped_array = img_array.reshape((h * w, d))
    
    if user_colors:
        centroids = np.array([hex_to_rgb(color) for color in user_colors if color])
        n_colors = len(centroids)
        kmeans = KMeans(n_clusters=n_colors, init=centroids, n_init=1)
    else:
        kmeans = KMeans(n_clusters=n_colors)
    
    kmeans.fit(reshaped_array)
    labels = kmeans.predict(reshaped_array)
    new_colors = kmeans.cluster_centers_.astype(np.uint8)
    new_img_array = new_colors[labels].reshape((h, w, d))
    
    return Image.fromarray(new_img_array), new_colors

def create_color_preview(colors):
    fig, ax = plt.subplots(1, len(colors), figsize=(2*len(colors), 2))
    if len(colors) == 1:
        ax = [ax]
    for i, color in enumerate(colors):
        ax[i].add_patch(plt.Rectangle((0, 0), 1, 1, fc=color/255))
        ax[i].axis('off')
        hex_color = rgb_to_hex(color)
        ax[i].text(0.5, -0.1, f'{hex_color}', ha='center', va='center', transform=ax[i].transAxes)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    color_preview = Image.open(buf)
    plt.close(fig)
    return color_preview

def apply_edge_preserve_smoothing(image, d=5, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(np.array(image), d, sigmaColor, sigmaSpace)

def sharpen_image(image):
    img_array = np.array(image)
    blurred = cv2.GaussianBlur(img_array, (0, 0), 3)
    sharpened = cv2.addWeighted(img_array, 1.5, blurred, -0.5, 0)
    return Image.fromarray(sharpened)

def generate_and_display(prompt, negative_prompt, num_images, guidance_scale, num_inference_steps, n_colors, image_format, width, height, use_user_colors, enable_tiling, *user_colors):
    try:
        images = generate_pattern(prompt, negative_prompt, num_images, guidance_scale, num_inference_steps, width, height, enable_tiling)
        output_paths = []
        color_preview = None
        display_images = []
        
        for i, img in enumerate(images):
            img_array = apply_edge_preserve_smoothing(img)
            img = Image.fromarray(img_array)
            
            if n_colors > 0 or (use_user_colors and any(user_colors)):
                if use_user_colors and any(user_colors):
                    selected_colors = [color for color in user_colors if color]
                    img, used_colors = limit_colors(img, len(selected_colors), selected_colors)
                else:
                    img, used_colors = limit_colors(img, n_colors)
                
                if i == 0:
                    color_preview = create_color_preview(used_colors)
            
            img = sharpen_image(img)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.{image_format.lower()}") as temp_file:
                img.save(temp_file, format=image_format)
                output_paths.append(temp_file.name)
                display_images.append(img)
        
        return display_images, output_paths, color_preview
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], [], None

with gr.Blocks() as iface:
    gr.Markdown("# Custom Pattern Generator (SDXL)")
    gr.Markdown("Generate custom patterns based on your input. Describe the pattern you want in detail in the prompt box.")
    
    prompt = gr.Textbox(label="Prompt", placeholder="Describe your pattern here (e.g., floral design with geometric shapes, art nouveau style)")
    negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Describe what you don't want in the image")
    enable_tiling = gr.Checkbox(label="Enable Tiling", value=False)
    
    with gr.Row():
        num_images = gr.Slider(1, 4, value=1, step=1, label="Number of Images to Generate")
        guidance_scale = gr.Slider(1, 20, value=7.5, label="Guidance Scale")
        num_inference_steps = gr.Slider(10, 100, value=50, step=1, label="Inference Steps")
        
    with gr.Row():
        n_colors = gr.Slider(0, 25, value=0, step=1, label="Number of Colors (0: unlimited)")
        use_user_colors = gr.Checkbox(label="Select Custom Colors", value=False)
        image_format = gr.Dropdown(choices=['PNG', 'JPEG', 'BMP', 'TIFF'], value='BMP', label="Image Format")
        
    with gr.Row():
        width = gr.Slider(512, 2048, value=1024, step=64, label="Width")
        height = gr.Slider(512, 2048, value=1024, step=64, label="Height")
    
    with gr.Row():
        color_pickers = [gr.ColorPicker(label=f"Color {i+1}", visible=False) for i in range(25)]
    
    def update_color_pickers(n, use_colors):
        return [gr.update(visible=(i < n and use_colors)) for i in range(25)]
    
    n_colors.change(
        update_color_pickers,
        inputs=[n_colors, use_user_colors],
        outputs=color_pickers
    )
    
    use_user_colors.change(
        update_color_pickers,
        inputs=[n_colors, use_user_colors],
        outputs=color_pickers
    )
    
    generate_btn = gr.Button("Generate Pattern")
    output_images = gr.Gallery(label="Generated Patterns")
    download_links = gr.File(label="Download Generated Patterns", file_count="multiple")
    color_preview = gr.Image(label="Used Colors")
    
    generate_btn.click(
        fn=generate_and_display,
        inputs=[prompt, negative_prompt, num_images, guidance_scale, num_inference_steps, n_colors, image_format, width, height, use_user_colors, enable_tiling] + color_pickers,
        outputs=[output_images, download_links, color_preview]
    )

if __name__ == "__main__":
    iface.launch()
