import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, clear_output
import time

%matplotlib inline

# ================== ENHANCED UPLOAD HANDLER ==================
class ImageUploader:
    def __init__(self):
        self.content_uploaded = False
        self.style_uploaded = False
        self.content_path = None
        self.style_path = None

    def upload_content(self, change):
        try:
            if not change['new']:
                return

            uploaded = change['owner']
            if not uploaded.value:
                raise ValueError("No file selected")

            self.content_path = next(iter(uploaded.value))
            with open(self.content_path, "wb") as f:
                f.write(uploaded.value[self.content_path]['content'])

            self.content_uploaded = True
            print("✓ Content image uploaded successfully")
            self.update_style_upload_status()

        except Exception as e:
            print(f"Error uploading content image: {str(e)}")
            self.content_uploaded = False

    def upload_style(self, change):
        if not self.content_uploaded:
            print("Please upload content image first")
            return

        try:
            if not change['new']:
                return

            uploaded = change['owner']
            if not uploaded.value:
                raise ValueError("No file selected")

            self.style_path = next(iter(uploaded.value))
            with open(self.style_path, "wb") as f:
                f.write(uploaded.value[self.style_path]['content'])

            self.style_uploaded = True
            print("✓ Style image uploaded successfully")

        except Exception as e:
            print(f"Error uploading style image: {str(e)}")
            self.style_uploaded = False

    def update_style_upload_status(self):
        if self.content_uploaded:
            print("You can now upload the style image")

# ================== STYLE TRANSFER FUNCTIONS ==================
# [Include all your existing style transfer functions here:
# preprocess_image, deprocess_image, show_images, get_model,
# gram_matrix, style_loss, content_loss, get_features, style_transfer]

# ================== MAIN INTERFACE ==================
def create_ui():
    uploader = ImageUploader()

    # Create widgets
    content_upload = widgets.FileUpload(accept='image/*', multiple=False)
    style_upload = widgets.FileUpload(accept='image/*', multiple=False)
    run_button = widgets.Button(description="Run Style Transfer", disabled=True)

    # Set up observers
    content_upload.observe(uploader.upload_content, names='value')
    style_upload.observe(uploader.upload_style, names='value')

    def check_ready():
        run_button.disabled = not (uploader.content_uploaded and uploader.style_uploaded)

    def on_run_click(b):
        clear_output()
        try:
            print("Starting style transfer...")
            show_images(uploader.content_path, uploader.style_path)
            final_image = style_transfer(uploader.content_path, uploader.style_path)

            plt.figure(figsize=(10, 10))
            plt.imshow(final_image)
            plt.axis('off')
            plt.title('Final Styled Image')
            plt.show()

            final_path = "styled_result.jpg"
            Image.fromarray(final_image).save(final_path)
            files.download(final_path)

        except Exception as e:
            print(f"Error during style transfer: {str(e)}")

    run_button.on_click(on_run_click)

    # Periodically check if both images are uploaded
    def update_button_status():
        while True:
            check_ready()
            time.sleep(1)

    import threading
    thread = threading.Thread(target=update_button_status)
    thread.daemon = True
    thread.start()

    # Display UI
    display(widgets.VBox([
        widgets.HTML("<h2>Neural Style Transfer</h2>"),
        widgets.HTML("<b>Step 1: Upload Content Image</b>"),
        content_upload,
        widgets.HTML("<b>Step 2: Upload Style Image</b>"),
        style_upload,
        run_button
    ]))

# Run the application
create_ui()


CODE2:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import time

# --- Optimized Configuration ---
TARGET_SIZE = (400, 400)  # Reduced from original
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
ITERATIONS = 250  # Reduced from 500
LEARNING_RATE = 5.0

# --- Pre-load Model ---
print("⏳ Loading VGG19 model...")
start_time = time.time()
vgg = vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
print(f"✅ Model loaded in {time.time()-start_time:.2f}s")

# --- Optimized Image Processing ---
def load_and_process_image(image_path, target_size=TARGET_SIZE):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

# --- Rest of your functions remain the same ---
# [Keep all your existing functions: deprocess_image, display_image,
#  build_model, gram_matrix, compute_losses]

# --- Optimized Style Transfer ---
def run_style_transfer(content_path, style_path, iterations=ITERATIONS):
    print("⏳ Loading and processing images...")
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)

    print("⏳ Building feature extraction model...")
    model = build_model()

    print("⏳ Computing targets...")
    content_target = model(content_image)[-1:]
    style_target = model(style_image)[:-1]

    generated = tf.Variable(content_image, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    best_img = None
    best_loss = float('inf')

    print("🚀 Starting style transfer...")
    start_time = time.time()

    for i in range(iterations):
        with tf.GradientTape() as tape:
            loss = compute_losses(model, generated, content_target, style_target,
                                 CONTENT_WEIGHT, STYLE_WEIGHT)
        grads = tape.gradient(loss, generated)
        optimizer.apply_gradients([(grads, generated)])
        generated.assign(tf.clip_by_value(generated, -1.5, 1.5))

        if loss < best_loss:
            best_loss = loss
            best_img = generated.numpy()

        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss.numpy():.2f}, Time elapsed: {time.time()-start_time:.2f}s")

    total_time = time.time() - start_time
    print(f"🎉 Style transfer completed in {total_time:.2f} seconds")
    return deprocess_image(best_img)

# --- Run with Progress Feedback ---
try:
    content_path = "/content/dog.jpg"
    style_path = "/content/kandinsky.jpg"

    print("🏁 Starting neural style transfer process")
    styled_result = run_style_transfer(content_path, style_path)

    plt.figure(figsize=(10, 10))
    plt.imshow(styled_result)
    plt.axis('off')
    plt.show()

    # Save result
    from PIL import Image
    Image.fromarray(styled_result).save("styled_output.jpg")
    print("💾 Styled image saved as 'styled_output.jpg'")

except Exception as e:
    print(f"❌ Error: {str(e)}")
