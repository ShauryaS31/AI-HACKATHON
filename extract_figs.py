import fitz  # PyMuPDF
import os
from PIL import Image
import io
import imagehash

# Define the path to the PDF file and the output folder
pdf_path = './pdf_documents/MAN_32-40_IMO_TierIII–Marine_.pdf'
output_folder = './extracted_figures'

# Path to the known image to ignore
ignore_image_path = './logo.jpg'

# Load the known image and compute its hash
ignore_image = Image.open(ignore_image_path)
ignore_image_hash = imagehash.phash(ignore_image)

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the PDF file
pdf_document = fitz.open(pdf_path)

# Initialize a counter for figure numbering
figure_counter = 1

# Regular expression to match titles with "Figure" followed by a number
import re
figure_title_pattern = re.compile(r'Figure\s+\d+', re.IGNORECASE)

# Threshold for image hash comparison
hash_threshold = 5

# Iterate over each page
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    # Get the text blocks from the page
    text_blocks = page.get_text("blocks")

    for block in text_blocks:
        text = block[4]
        if figure_title_pattern.search(text):
            # Extract figure title
            figure_title = text.strip()

            # Search for the image associated with the figure
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                try:
                    # Compute the hash of the current image
                    image = Image.open(io.BytesIO(image_bytes))
                    image_hash = imagehash.phash(image)

                    # Check if the image hash is similar to the known image hash
                    if abs(image_hash - ignore_image_hash) <= hash_threshold:
                        continue

                    if image_ext.lower() == "jpx":
                        image = image.convert("RGB")
                        image_path = os.path.join(output_folder, f"figure_{figure_counter}.jpg")
                        image.save(image_path, "JPEG")
                    else:
                        image_path = os.path.join(output_folder, f"figure_{figure_counter}.{image_ext}")
                        with open(image_path, "wb") as image_file:
                            image_file.write(image_bytes)
                except OSError as e:
                    print(f"Error converting image {figure_counter}: {e}")
                    continue
                
                # Save the title in a text file
                title_path = os.path.join(output_folder, f"figure_{figure_counter}_title.txt")
                with open(title_path, "w") as title_file:
                    title_file.write(figure_title)

                figure_counter += 1

# Close the PDF document
pdf_document.close()

print("Figures and titles extracted successfully.")
