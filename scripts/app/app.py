import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import base64
import io
from PIL import Image
import numpy as np
import torch
from visualizations import load_model, config, transform_batch
import ocl
from llm2vec import LLM2Vec
import os
import pickle

batch_idx = 54
n_slots = 4

# Initialize the app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Load the model
datamodule, model = load_model(config, n_slots=n_slots)

#save all the images in the dataloader in a folder called images
# os.makedirs("./images", exist_ok=True)
# for i, batch in enumerate(datamodule.val_dataloader()):
#     input_image = batch['image'][0]
#     input_image = input_image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
#     input_image = input_image.cpu().numpy()
    
#     # Denormalize the image
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     input_image = std * input_image + mean
    
#     # Clip values to [0, 1] range and convert to uint8
#     input_image = (np.clip(input_image, 0, 1) * 255).astype(np.uint8)
    
#     input_image = Image.fromarray(input_image)
    
#     input_image.save(f"./images/{i}.png")

# Get the first image from the dataloader
for i, batch in enumerate(datamodule.val_dataloader()):
    if i == batch_idx:
        input_image = batch['image'][0]
        #torch.Size([3, 224, 224])
        input_image = input_image[torch.tensor([2, 1, 0]), :, :]
        break

# Convert the image tensor to a numpy array
input_image_np = input_image.permute(1, 2, 0).cpu().numpy()

# Denormalize the image
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
input_image_np = std * input_image_np + mean

# Clip values to [0, 1] range
input_image_np = np.clip(input_image_np, 0, 1)*255
# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(id='image-plot', style={'width': '100%', 'height': '600px'}),
    html.Div([
        dcc.Input(id='object-name', type='text', placeholder='Enter object name'),
        dcc.Input(id='slot', type='number', placeholder='Enter slot'),
        html.Button('Add Point', id='add-point-button', n_clicks=0),
        html.Button('Generate Masks', id='generate-masks-button', n_clicks=0),
        html.Button('Clear Points', id='clear-points-button', n_clicks=0),
    ]),
    html.Div(id='points-list'),
    html.Div(id='output-masks')
])

# Add a new callback to manage points
@app.callback(
    Output('points-list', 'children'),
    Input('add-point-button', 'n_clicks'),
    Input('clear-points-button', 'n_clicks'),
    State('image-plot', 'clickData'),
    State('object-name', 'value'),
    State('slot', 'value'),
    State('points-list', 'children'),
)
def manage_points(add_clicks, clear_clicks, clickData, object_name, slot, existing_points):
    ctx = dash.callback_context
    if not ctx.triggered:
        return existing_points or []
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'clear-points-button':
        return []
    
    if button_id == 'add-point-button' and clickData and object_name and slot is not None:
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        new_point = html.Div(f"Object: {object_name}, Slot: {slot}, X: {x:.2f}, Y: {y:.2f}")
        return (existing_points or []) + [new_point]
    
    return existing_points or []

# Update the graph callback
@app.callback(
    Output('image-plot', 'figure'),
    Input('points-list', 'children')
)
def update_graph(points_list):
    fig = go.Figure()

    # Add the input image
    fig.add_trace(go.Image(z=input_image_np))

    # Add points if they exist
    if points_list:
        for point in points_list:
            point_text = point['props']['children']
            _, _, x_str, y_str = point_text.split(', ')
            x = float(x_str.split(': ')[1])
            y = float(y_str.split(': ')[1])
            object_name = point_text.split(', ')[0].split(': ')[1]
            
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=10, color='red')))
            fig.add_annotation(x=x, y=y, text=object_name, showarrow=False, yshift=20)

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )

    return fig

# Update the generate_masks callback
@app.callback(
    Output('output-masks', 'children'),
    Input('generate-masks-button', 'n_clicks'),
    State('points-list', 'children')
)
def generate_masks(n_clicks, points_list):
    slots_nums = []
    if n_clicks > 0 and points_list:
        change_queries = {}
        for point in points_list:
            point_text = point['props']['children']
            object_name, slot_str, x_str, y_str = point_text.split(', ')
            object_name = object_name.split(': ')[1]
            slot = int(slot_str.split(': ')[1])
            slots_nums.append(slot)
            x = float(x_str.split(': ')[1]) / input_image_np.shape[1]
            y = float(y_str.split(': ')[1]) / input_image_np.shape[0]
            change_queries[slot] = {"name": object_name, "coors": [x, y]}
        
        # Prepare the batch with the new points and object names
        for i, batch in enumerate(datamodule.val_dataloader()):
            if i == batch_idx:
                break

        batch = transform_batch(batch, change_queries)
        
        # Move batch to the correct device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(model.device)
        
        # Generate output
        with torch.no_grad():
            output = model(batch)
        
        # Get masks and input image
        masks = output['object_decoder'].masks_as_image[0]
        input_image = output['input']["image"][0]
        pred_masks_matched = ocl.visualizations.Segmentation(
            denormalization=ocl.preprocessing.Denormalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
        vis = pred_masks_matched(
            image=output['input']["image"],
            mask=output['object_decoder'].masks_as_image
        )
        masks_vis = vis.img_tensor.permute(1, 2, 0).cpu().numpy()
        # Ensure input_image is on CPU
        input_image = input_image.cpu()

        # Denormalize the input image
        denorm = ocl.preprocessing.Denormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # Move denorm parameters to CPU
        denorm.mean = denorm.mean.cpu()
        denorm.std = denorm.std.cpu()
        input_image_denorm = denorm(input_image)
        input_image_denorm = input_image_denorm[torch.tensor([2, 1, 0]), :, :]
        # Convert masks to images and encode them
        mask_images = []
        for i in range(masks.shape[0]):
            # Move mask to CPU and apply to input image
            mask = masks[i].cpu()
            masked_image = input_image_denorm * mask.unsqueeze(0)
            
            # Convert to PIL Image
            masked_image_pil = Image.fromarray((masked_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            
            # Encode image
            buffered = io.BytesIO()
            masked_image_pil.save(buffered, format="PNG")
            mask_encoded = base64.b64encode(buffered.getvalue()).decode()
            
            # Add to list of images
            mask_images.append(html.Img(src=f'data:image/png;base64,{mask_encoded}', 
                                        style={'width': '200px', 'height': '200px', 'margin': '5px'}))
        
        # Add original image
        original_image_pil = Image.fromarray((input_image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        masks_vis_pil = Image.fromarray(masks_vis)
        buffered = io.BytesIO()
        buffered2 = io.BytesIO()
        original_image_pil.save(buffered, format="PNG")
        masks_vis_pil.save(buffered2, format="PNG")
        original_encoded = base64.b64encode(buffered.getvalue()).decode()
        masks_vis_encoded = base64.b64encode(buffered2.getvalue()).decode()
        slot_masks = [mask for i, mask in enumerate(mask_images) if i in slots_nums]
        original_image = html.Img(src=f'data:image/png;base64,{original_encoded}', 
                                  style={'width': '200px', 'height': '200px', 'margin': '5px'})
        masks_vis_pil = html.Img(src=f'data:image/png;base64,{masks_vis_encoded}', 
                                  style={'width': '200px', 'height': '200px', 'margin': '5px'})
        return html.Div([
            html.Div([
                html.Div([
                    html.H3("Original Image"),
                    original_image
                ], style={'display': 'inline-block', 'margin-right': '20px'}),
                
                html.Div([
                    html.H3("Masks"),
                    masks_vis_pil
                ], style={'display': 'inline-block'}),
                html.Div([
                    html.H3("Masked Image Parts"),
                    html.Div(slot_masks)
                ], style={'display': 'inline-block'})
            ], style={'white-space': 'nowrap'}),
            html.H3("Masked Image Parts"),
            html.Div(mask_images)
        ])
    
    return "Click 'Generate Masks' after adding points and object names."




if __name__ == '__main__':
    app.run_server(debug=True)
