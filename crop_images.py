from PIL import Image

def crop_image(image_path, bbox):
 
    with Image.open(image_path) as img:
        # Calculate the bottom right corner of the bounding box
        left, top, right, bottom = bbox
        
        
        # Crop the image
        cropped_image = img.crop((left, top, right, bottom))
        return cropped_image

# Example usage
# bbox = (50, 50, 100, 100)  # Example bounding box coordinates
# cropped_img = crop_image("path_to_your_image.jpg", bbox)
# cropped_img.show()  # This will display the cropped image
