from PIL import Image

def image_overlay(b_path,o_path,position):
    background = Image.open(b_path)

    # Load the object image
    object_image = Image.open(o_path)

    # Convert the object image to have an alpha channel (transparency)
    object_image = object_image.convert("RGBA")

    # Get the data of the object image
    datas = object_image.getdata()

    # Replace white pixels with transparent pixels
    new_data = []
    for item in datas:
        # Change all white (also shades of whites)
        # to transparent
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    object_image.putdata(new_data)

    # Get the size of the background image
    bg_width, bg_height = background.size

    # Define the position where the object will be placed
    # position = ((bg_width - object_image.width) // 2, (bg_height - object_image.height) // 2)
    
    # Paste the object image onto the background image
    background.paste(object_image, position, object_image)
    
    return(background)

# # Save the resulting image
# background.save('result_image3.png')

# python run2.py --image /home/vaibhavsharma/Desktop/rfp/unstructured_excels/segment_anything/inputs/office_chair.jpeg --object office_chair --output /home/vaibhavsharma/Desktop/rfp/unstructured_excels/segment_anything/task2_outputs/office_chair.jpg
