# NEURAL-STYLE-TRANSFER

*COMPANY: CODITECH IT SOLUTIONS

*NAME: SRIKAR BHARADWAJ

*INTERN ID: CT08DM833

*DOMAIN: ARTIFICIAL INTELLIGENCE

*DURATION: 8 WEEKS

*MENTOR: NEELA SANTOSH

*DESCRIPTION:
🎨 Neural Style Transfer in Simple Terms
Neural Style Transfer (NST) is a cool technique where you take two images — a content image (like a photo of your face or a cityscape) and a style image (like a famous painting) — and blend them together. The goal is to keep the shapes and layout from the content image, but paint it in the artistic style of the style image. So you could, for example, turn a selfie into a "Van Gogh-style painting!"

🧠 How It Works
Your code uses deep learning, specifically a pre-trained model called VGG19, which is great at understanding images. It looks at both images and pulls out:

Content features: the structure, layout, and objects in the photo.

Style features: the colors, textures, and patterns in the artwork.

It then creates a new image that tries to match the content of the first and the style of the second.

🖥️ What This Code Does
This project is set up as an interactive interface in Google Colab, making it beginner-friendly and fun to use. Here's a breakdown of what each part does:

ImageUploader Class
This helps manage the two images: one for content and one for style. It lets the user upload images and gives messages like "✓ Content image uploaded successfully" so you know things are working.

UI with Widgets
It creates an interface with buttons and upload boxes using ipywidgets. It walks you through:

Uploading a content image

Uploading a style image

Clicking a button to start the style transfer

Running the Style Transfer
Once both images are uploaded, you can click “Run Style Transfer”. The model processes the images and outputs a new one, combining the best of both. It shows you the final result and even lets you download it as a file.

🧩 What’s Missing?
Right now, this code is just the interface. It still needs the style transfer engine — which includes functions like:

preprocess_image()

style_transfer()

show_images()

These will actually load, process, and combine the images. Without them, the "Run" button won’t do the transformation.

🧑‍🎨 Why This is Cool
This project lets you create art using AI — even if you’re not a coder. You can upload your own photos and favorite artworks and watch magic happen in seconds.

OUTPUT: ![Image](https://github.com/user-attachments/assets/bc016494-c4ce-486b-9d93-b1a8867b55eb)

![Image](https://github.com/user-attachments/assets/c115c121-d4f5-48cd-b585-382b9fad0cad)

![Image](https://github.com/user-attachments/assets/0d1444df-e807-486d-9b64-d168ec9ae32e)
