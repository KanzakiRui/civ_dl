import os
import json
import requests
import gradio as gr
import time
import urllib.request
import shutil
import datetime
from modules import script_callbacks, shared, ui_components
from fastapi import FastAPI, Request, Response

# Define the extension's configuration path
EXTENSION_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(EXTENSION_DIR, "config.json")

# Default configuration
DEFAULT_CONFIG = {
    "api_token": "",
    "download_path": os.path.join(shared.models_path, "Stable-diffusion"),
    "preview_count": 9,
    "timeout": 60,
    "default_model_type": "Checkpoint",
    "save_preview_grid": True,
    "sort_by": "Most Downloaded",
    "time_period": "All Time",
    "base_model": "All",
    "search_type": "Models",
    "token_based_auth": False
}

# Model types and their corresponding directories
MODEL_TYPES = {
    "Checkpoint": os.path.join(shared.models_path, "Stable-diffusion"),
    "LORA": os.path.join(shared.models_path, "Lora"),
    "Hypernetwork": os.path.join(shared.models_path, "hypernetworks"),
    "TextualInversion": os.path.join(shared.models_path, "embeddings"),
    "AestheticGradient": os.path.join(shared.models_path, "aesthetic_embeddings"),
    "VAE": os.path.join(shared.models_path, "VAE"),
    "Controlnet": os.path.join(shared.models_path, "ControlNet"),
    "Pose": os.path.join(shared.models_path, "Poses")
}

# Sort options
SORT_OPTIONS = [
    "Most Downloaded",
    "Highest Rated",
    "Newest",
    "Most Liked",
    "Most Discussed"
]

# Time period options
TIME_PERIODS = [
    "All Time",
    "Year",
    "Month",
    "Week",
    "Day"
]

# Base model options
BASE_MODELS = [
    "All",
    "SD 1.5",
    "SD 2.0",
    "SD 2.1",
    "SD XL",
    "SDXL Turbo",
    "SVD",
    "Other"
]

# Search type options
SEARCH_TYPES = [
    "Models",
    "Images",
    "Articles"
]

# Map API parameter names
SORT_BY_MAP = {
    "Most Downloaded": "downloadCount",
    "Highest Rated": "rating",
    "Newest": "createdAt",
    "Most Liked": "likes",
    "Most Discussed": "commentCount"
}

TIME_PERIOD_MAP = {
    "All Time": None,
    "Year": "year",
    "Month": "month",
    "Week": "week",
    "Day": "day"
}

BASE_MODEL_MAP = {
    "All": None,
    "SD 1.5": "SD 1.5",
    "SD 2.0": "SD 2.0",
    "SD 2.1": "SD 2.1",
    "SD XL": "SDXL",
    "SDXL Turbo": "SDXL Turbo",
    "SVD": "SVD",
    "Other": "Other"
}

# Load configuration
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            # Ensure all default values are present
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
    else:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG.copy()

# Save configuration
def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

# Get configuration
def get_config():
    return load_config()

# Get headers
def get_headers():
    config = get_config()
    headers = {
        "User-Agent": "CivitaiDownloaderExtension/1.1",
        "Accept": "application/json"
    }
    
    # Use token-based authentication if enabled
    if config["token_based_auth"] and config["api_token"]:
        headers["Authorization"] = f"Bearer {config['api_token']}"
    # Otherwise use the legacy method
    elif config["api_token"]:
        headers["Authorization"] = f"Token {config['api_token']}"
        
    return headers

def search_models(query, page=1, limit=20, nsfw=False, types=None, sort_by=None, period=None, baseModel=None, search_type="Models"):
    params = {
        "query": query,
        "page": page,
        "limit": limit,
        "nsfw": str(nsfw).lower()
    }
    
    if types:
        params["types"] = ",".join(types)
    
    if sort_by:
        params["sort"] = sort_by
    
    if period:
        params["period"] = period
        
    if baseModel:
        params["baseModel"] = baseModel
    
    # Choose the appropriate API endpoint based on search type
    endpoint = "models"
    if search_type == "Images":
        endpoint = "images"
    elif search_type == "Articles":
        endpoint = "articles"
    
    response = requests.get(
        f"https://civitai.com/api/v1/{endpoint}", 
        params=params, 
        headers=get_headers(), 
        timeout=get_config()["timeout"]
    )
    return response.json()

def get_model_details(model_id):
    response = requests.get(
        f"https://civitai.com/api/v1/models/{model_id}", 
        headers=get_headers(), 
        timeout=get_config()["timeout"]
    )
    return response.json()

def get_model_versions(model_id):
    response = requests.get(
        f"https://civitai.com/api/v1/models/{model_id}/versions", 
        headers=get_headers(), 
        timeout=get_config()["timeout"]
    )
    return response.json()

def download_file(url, destination):
    # Make sure the directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    config = get_config()
    
    # Set up headers for download
    headers = {}
    if config["token_based_auth"] and config["api_token"]:
        headers["Authorization"] = f"Bearer {config['api_token']}"
    elif config["api_token"]:
        headers["Authorization"] = f"Token {config['api_token']}"
    
    # Open a stream to the URL
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        # Get total file size
        file_size = int(response.headers.get('Content-Length', 0))
        
        # Download and write to file
        with open(destination, 'wb') as f:
            shutil.copyfileobj(response, f)
    
    return os.path.exists(destination)

def download_images(images, folder):
    os.makedirs(folder, exist_ok=True)
    downloaded_images = []
    
    for i, img in enumerate(images):
        url = img.get("url")
        if url:
            ext = os.path.splitext(url)[1]
            filename = f"preview_{i}{ext}"
            filepath = os.path.join(folder, filename)
            
            try:
                download_file(url, filepath)
                downloaded_images.append(filepath)
            except Exception as e:
                print(f"Error downloading image {i}: {e}")
    
    return downloaded_images

def create_preview_grid(images, output_path, grid_size=3):
    try:
        from PIL import Image
        import math
        
        # Limit to the first n images based on grid size squared
        num_images = min(len(images), grid_size * grid_size)
        images = images[:num_images]
        
        if not images:
            return None
        
        # Open all images
        opened_images = [Image.open(img) for img in images]
        
        # Calculate grid dimensions
        cols = min(grid_size, len(opened_images))
        rows = math.ceil(len(opened_images) / cols)
        
        # Get first image dimensions
        width, height = opened_images[0].size
        
        # Create new image
        grid_img = Image.new('RGB', (width * cols, height * rows))
        
        # Paste images into grid
        for i, img in enumerate(opened_images):
            x = (i % cols) * width
            y = (i // cols) * height
            grid_img.paste(img, (x, y))
        
        # Save the grid image
        grid_img.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error creating preview grid: {e}")
        return None

# UI Components
def on_ui_tabs():
    config = get_config()
    
    with gr.Blocks(analytics_enabled=False) as civitai_interface:
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Box():
                    gr.HTML("<h2>Civitai Downloader</h2>")
                    with gr.Row():
                        search_term = gr.Textbox(label="Search", placeholder="Search models...")
                        search_button = gr.Button("Search", variant="primary")
                    
                    with gr.Accordion("Search Options", open=True):
                        with gr.Row():
                            search_type_dropdown = gr.Dropdown(
                                label="Search Type",
                                choices=SEARCH_TYPES,
                                value=config["search_type"]
                            )
                            sort_by_dropdown = gr.Dropdown(
                                label="Sort By",
                                choices=SORT_OPTIONS,
                                value=config["sort_by"]
                            )
                        
                        with gr.Row():
                            time_period_dropdown = gr.Dropdown(
                                label="Time Period",
                                choices=TIME_PERIODS,
                                value=config["time_period"]
                            )
                            base_model_dropdown = gr.Dropdown(
                                label="Base Model",
                                choices=BASE_MODELS,
                                value=config["base_model"]
                            )
                            
                        with gr.Row():
                            model_type_dropdown = gr.Dropdown(
                                label="Filter by Type",
                                choices=list(MODEL_TYPES.keys()),
                                value=config["default_model_type"],
                                multiselect=True
                            )
                            include_nsfw = gr.Checkbox(label="Include NSFW", value=False)
                    
                    search_results = gr.Dataframe(
                        headers=["ID", "Name", "Type", "Downloads", "Rating"],
                        datatype=["number", "str", "str", "number", "number"],
                        col_count=(5, "fixed"),
                        interactive=False,
                        label="Search Results"
                    )
                    
                    with gr.Row():
                        prev_page = gr.Button("Previous Page")
                        page_info = gr.Textbox(value="Page 1", label="", interactive=False)
                        next_page = gr.Button("Next Page")
            
            with gr.Column(scale=2):
                with gr.Box():
                    gr.HTML("<h3>Model Details</h3>")
                    model_name = gr.Textbox(label="Name", interactive=False)
                    model_description = gr.Textbox(label="Description", interactive=False, lines=3)
                    model_info = gr.JSON(label="Additional Information", interactive=False)
                    
                    with gr.Box():
                        gr.HTML("<h4>Select Version</h4>")
                        model_versions = gr.Dropdown(label="Available Versions", choices=[], interactive=True)
                    
                    with gr.Box():
                        gr.HTML("<h4>Preview Images</h4>")
                        preview_gallery = gr.Gallery(label="Preview Images", show_label=False, object_fit="contain", height=300)
                    
                    with gr.Row():
                        download_btn = gr.Button("Download Selected Version", variant="primary")
                        download_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Settings"):
            with gr.Accordion("API Configuration", open=True):
                with gr.Row():
                    api_token = gr.Textbox(
                        label="Civitai API Token (optional)",
                        value=config["api_token"],
                        placeholder="Enter your API token for accessing restricted content",
                        type="password"
                    )
                    token_based_auth = gr.Checkbox(
                        label="Use New Bearer Token Authentication",
                        value=config.get("token_based_auth", False),
                        info="Enable this for newer Civitai API (Bearer token)"
                    )

            with gr.Accordion("Download Settings", open=True):
                download_path = gr.Textbox(
                    label="Default Download Path",
                    value=config["download_path"],
                    placeholder="Enter default download path"
                )
                
                default_model_type = gr.Dropdown(
                    label="Default Model Type",
                    choices=list(MODEL_TYPES.keys()),
                    value=config["default_model_type"]
                )
                
                save_preview_grid = gr.Checkbox(
                    label="Save Preview Grid with Model",
                    value=config["save_preview_grid"]
                )
            
            with gr.Accordion("Display Settings", open=True):
                preview_count = gr.Slider(
                    label="Preview Image Count",
                    minimum=1,
                    maximum=20,
                    value=config["preview_count"],
                    step=1
                )
                
                timeout = gr.Slider(
                    label="Request Timeout (seconds)",
                    minimum=10,
                    maximum=300,
                    value=config["timeout"],
                    step=5
                )
            
            with gr.Accordion("Default Search Options", open=True):
                default_search_type = gr.Dropdown(
                    label="Default Search Type",
                    choices=SEARCH_TYPES,
                    value=config["search_type"]
                )
                
                default_sort_by = gr.Dropdown(
                    label="Default Sort Order",
                    choices=SORT_OPTIONS,
                    value=config["sort_by"]
                )
                
                default_time_period = gr.Dropdown(
                    label="Default Time Period",
                    choices=TIME_PERIODS,
                    value=config["time_period"]
                )
                
                default_base_model = gr.Dropdown(
                    label="Default Base Model",
                    choices=BASE_MODELS,
                    value=config["base_model"]
                )
                
            save_settings_btn = gr.Button("Save Settings", variant="primary")
            settings_status = gr.Textbox(label="Status", interactive=False)
        
        # State variables
        current_page = gr.State(1)
        current_model_id = gr.State(None)
        current_model_data = gr.State(None)
        current_versions_data = gr.State(None)
        
        # Event handlers for search functionality
        def perform_search(search_query, page, model_types, nsfw):
            try:
                types = model_types if isinstance(model_types, list) else [model_types]
                results = search_models(search_query, page=page, nsfw=nsfw, types=types)
                
                models_data = []
                for model in results.get("items", []):
                    models_data.append([
                        model.get("id"),
                        model.get("name"),
                        model.get("type"),
                        model.get("stats", {}).get("downloadCount", 0),
                        model.get("stats", {}).get("rating", 0)
                    ])
                
                page_text = f"Page {page} of {results.get('metadata', {}).get('totalPages', 1)}"
                return models_data, page_text
            except Exception as e:
                return [], f"Error: {str(e)}"
        
        def search_and_update(search_query, page, model_types, nsfw):
            results, page_info_text = perform_search(search_query, page, model_types, nsfw)
            return results, page_info_text, page
        
        search_button.click(
            search_and_update,
            inputs=[search_term, gr.State(1), model_type_dropdown, include_nsfw],
            outputs=[search_results, page_info, current_page]
        )
        
        def change_page(direction, current_page, search_query, model_types, nsfw):
            new_page = max(1, current_page + direction)
            results, page_info_text = perform_search(search_query, new_page, model_types, nsfw)
            return results, page_info_text, new_page
        
        prev_page.click(
            change_page,
            inputs=[gr.State(-1), current_page, search_term, model_type_dropdown, include_nsfw],
            outputs=[search_results, page_info, current_page]
        )
        
        next_page.click(
            change_page,
            inputs=[gr.State(1), current_page, search_term, model_type_dropdown, include_nsfw],
            outputs=[search_results, page_info, current_page]
        )
        
        # Event handlers for model details
        def load_model_details(evt: gr.SelectData, results):
            try:
                row_index = evt.index[0]
                model_id = results[row_index][0]
                
                # Get model details and versions
                model_data = get_model_details(model_id)
                versions_data = get_model_versions(model_id)
                
                # Update model info
                name = model_data.get("name", "")
                description = model_data.get("description", "")
                
                # Extract additional info
                info = {
                    "Creator": model_data.get("creator", {}).get("username", "Unknown"),
                    "Type": model_data.get("type", ""),
                    "NSFW": model_data.get("nsfw", False),
                    "Tags": [tag.get("name") for tag in model_data.get("tags", [])],
                    "Downloads": model_data.get("stats", {}).get("downloadCount", 0),
                    "Rating": model_data.get("stats", {}).get("rating", 0),
                }
                
                # Process versions for dropdown
                versions = []
                versions_map = {}
                
                for version in versions_data.get("items", []):
                    version_name = f"{version.get('name')} (Size: {version.get('fileSize') / (1024*1024):.1f} MB)"
                    versions.append(version_name)
                    versions_map[version_name] = version
                
                # Get preview images from the first version
                preview_images = []
                first_version = versions_data.get("items", [])
                
                if first_version:
                    preview_images = first_version[0].get("images", [])
                    preview_urls = [img.get("url") for img in preview_images[:int(get_config()["preview_count"])]]
                
                return (
                    model_id, 
                    name, 
                    description, 
                    info, 
                    versions, 
                    preview_urls, 
                    model_data, 
                    versions_data
                )
            except Exception as e:
                return None, f"Error: {str(e)}", "", {}, [], [], None, None
        
        search_results.select(
            load_model_details,
            inputs=[search_results],
            outputs=[
                current_model_id, 
                model_name, 
                model_description, 
                model_info, 
                model_versions, 
                preview_gallery, 
                current_model_data, 
                current_versions_data
            ]
        )
        
        # Event handler for version selection
        def on_version_select(version_name, versions_data):
            if not version_name or not versions_data:
                return []
            
            for version in versions_data.get("items", []):
                if version_name.startswith(version.get("name")):
                    preview_images = version.get("images", [])
                    preview_urls = [img.get("url") for img in preview_images[:int(get_config()["preview_count"])]]
                    return preview_urls
            
            return []
        
        model_versions.change(
            on_version_select,
            inputs=[model_versions, current_versions_data],
            outputs=[preview_gallery]
        )
        
        # Event handler for download
        def download_model(model_id, version_name, model_data, versions_data):
            if not model_id or not version_name or not model_data or not versions_data:
                return "Please select a model and version first."
            
            try:
                config = get_config()
                
                # Find the selected version
                selected_version = None
                for version in versions_data.get("items", []):
                    if version_name.startswith(version.get("name")):
                        selected_version = version
                        break
                
                if not selected_version:
                    return "Version not found."
                
                # Get download URL
                download_url = selected_version.get("downloadUrl")
                if not download_url:
                    return "Download URL not available."
                
                # Determine file path
                model_type = model_data.get("type")
                base_dir = MODEL_TYPES.get(model_type, config["download_path"])
                
                # Create sanitized filename
                filename = selected_version.get("files", [{}])[0].get("name")
                if not filename:
                    filename = f"{model_data.get('name')}_{selected_version.get('name')}.safetensors"
                    filename = filename.replace(" ", "_").replace(":", "_").replace("/", "_")
                
                # Full path for the file
                file_path = os.path.join(base_dir, filename)
                
                # Download the model file
                success = download_file(download_url, file_path)
                
                if not success:
                    return f"Failed to download {filename}."
                
                # Save preview images if configured
                if config["save_preview_grid"] and selected_version.get("images"):
                    # Create a directory for previews
                    preview_dir = os.path.join(
                        base_dir, 
                        "previews", 
                        os.path.splitext(filename)[0]
                    )
                    
                    # Download preview images
                    preview_images = selected_version.get("images", [])[:int(config["preview_count"])]
                    downloaded_images = download_images(preview_images, preview_dir)
                    
                    # Create and save preview grid
                    if downloaded_images:
                        grid_path = os.path.join(base_dir, f"{os.path.splitext(filename)[0]}_preview.jpg")
                        create_preview_grid(downloaded_images, grid_path)
                
                return f"Successfully downloaded {filename} to {base_dir}"
            except Exception as e:
                return f"Error during download: {str(e)}"
        
        download_btn.click(
            download_model,
            inputs=[current_model_id, model_versions, current_model_data, current_versions_data],
            outputs=[download_status]
        )
        
        # Settings handlers
        def save_settings(token, path, preview_count, timeout, model_type, save_grid):
            try:
                config = get_config()
                config["api_token"] = token
                config["download_path"] = path
                config["preview_count"] = int(preview_count)
                config["timeout"] = int(timeout)
                config["default_model_type"] = model_type
                config["save_preview_grid"] = save_grid
                save_config(config)
                return "Settings saved successfully."
            except Exception as e:
                return f"Error saving settings: {str(e)}"
        
        save_settings_btn.click(
            save_settings,
            inputs=[api_token, download_path, preview_count, timeout, default_model_type, save_preview_grid],
            outputs=[settings_status]
        )
        
    return [(civitai_interface, "Civitai Downloader", "civitai_downloader")]

# Register the extension
script_callbacks.on_ui_tabs(on_ui_tabs)
