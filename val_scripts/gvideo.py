from habitat.utils.visualizations.utils import images_to_video
def gvideo(imgs ,
           path = '/home/getuanhui/project/sound-spaces/yz/soundspaces_data/env_data/video', 
           name = 'default'):
    """
    if you use pickle to load the collect data , you will get the data[0][0]['camera']
    path: you can set a new path and suggest input a abusolut path
    name is the video name
    """
    images_to_video(imgs , path , name)