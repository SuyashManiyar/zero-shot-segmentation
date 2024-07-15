from PIL import Image

def flip_image(img_path,angle):
    image_path = img_path# Replace with your image path
    image = Image.open(image_path)

    # Rotate the image
    rotated_image = image.rotate(angle)  # Rotate by 90 degrees (you can change the angle as needed)

    # Flip the image horizontally
    horizontally_flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Flip the image vertically
    vertically_flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Save the images
    # rotated_image.save('rotated_image.jpg')
    # horizontally_flipped_image.save('horizontally_flipped_image.jpg')
    # vertically_flipped_image.save('vertically_flipped_image.jpg')

    # Show the images (optional)
    # rotated_image.show()
    # horizontally_flipped_image.show()
    # vertically_flipped_image.show()
    
    return rotated_image,horizontally_flipped_image,vertically_flipped_image
