# Fast Style Transfer for Arbitrary Styles
Based on the model code in magenta and the publication:

Exploring the structure of a real-time, arbitrary neural artistic stylization network. Golnaz Ghiasi, Honglak Lee, Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens, Proceedings of the British Machine Vision Conference (BMVC), 2017.

## Demonstrate image stylization
<img width="721" alt="Screenshot at Dec 02 19-27-20" src="https://user-images.githubusercontent.com/94981693/144456832-5b5cc638-ef42-49a8-acfa-ec682411bbcb.png">

### Import TF Hub module

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

The signature of this hub module for image stylization is:

outputs = hub_module(content_image, style_image)
stylized_image = outputs[0]

Where content_image, style_image, and stylized_image are expected to be 4-D Tensors with shapes [batch_size, image_height, image_width, 3].
In the current example we provide only single images and therefore the batch dimension is 1, but one can use the same module to process more images at the same time.
The input and output values of the images should be in the range [0, 1].
The shapes of content and style image don't have to match. Output image shape is the same as the content image shape.

## Specify the main content image and the style you want to use

content_name = 'sea_turtle'  # @param ['sea_turtle', 'tuebingen', 'grace_hopper']
style_name = 'munch_scream'  # @param ['kanagawa_great_wave', 'kandinsky_composition_7', 'hubble_pillars_of_creation', 'van_gogh_starry_night', 'turner_nantes', 'munch_scream', 'picasso_demoiselles_avignon', 'picasso_violin', 'picasso_bottle_of_rum', 'fire', 'derkovits_woman_head', 'amadeo_style_life', 'derkovtis_talig', 'amadeo_cardoso']

stylized_image = hub_module(tf.constant(content_images[content_name]),
                            tf.constant(style_images[style_name]))[0]

show_n([content_images[content_name], style_images[style_name], stylized_image],
       titles=['Original content image', 'Style image', 'Stylized image'])       


![image](https://user-images.githubusercontent.com/94981693/144458955-8689a372-d8f0-4ddd-9d48-ce882d49e2be.png)
