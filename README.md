# fish_tracking
##Expected behaviour
If everything works *properly* this should take a pretrained YOLO classification model and a video of fish swimming in an arena --> get the positions of the heads, tails, and centroids of each fish, as well as its angular velocity, speed,and midline offset. It should also give the times that the light stimulus is on/off and store all this information in a Excel file with one sheet per fish.
##Known failure cases
-When the fish doesn't move in the whole video. This causes it to get masked out and become invisible.
-When the fish is too close to the the rim. The rim, like the fish, is dark, and can mess up the skeletonization if it gets in the bounding box.
-When the belly glow is too bright. I have figured out some ways to 'patch' it: Gaussian blur, erosion and dilation, flood fill... but none of these solution seems to be perfect

Nevertheless the trajectories are usually mostly accurate (especially in the good lighting, cleaner background examples) and some smoothing of the overall trajectory may help with the accuracy.
