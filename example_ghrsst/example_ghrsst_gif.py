import imageio.v2 as imageio
import os

#MM = '12'

#with imageio.get_writer(f'example_ghrsst/gifs/example_ghrsst_{MM}.gif', mode='I') as writer:
#    for filename in os.listdir('example_ghrsst/images/'+MM):
#        image = imageio.imread('example_ghrsst/images/'+MM+'/'+filename)
#        writer.append_data(image)

with imageio.get_writer('example_ghrsst/gifs/example_ghrsst.gif', mode='I') as writer:
    for folder in os.listdir('example_ghrsst/images'):
        for file in os.listdir('example_ghrsst/images/'+folder):
            image = imageio.imread('example_ghrsst/images/'+folder+'/'+file)
            writer.append_data(image)