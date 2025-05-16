import launch
import os

# Check gradio version for compatibility
launch.check_versions()
print("Launching Civitai Downloader extension...")

# Create extension directory structure
def create_directories():
    extension_dir = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(os.path.join(extension_dir, "previews"), exist_ok=True)
    print(f"Extension directories created at {extension_dir}")

if __name__ == "__main__":
    create_directories()
