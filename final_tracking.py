import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline, splrep, splev
from skimage.morphology import skeletonize,  flood_fill, flood
from skimage.measure import label, regionprops, regionprops_table, find_contours
import cv2
from ultralytics import YOLO
from dfs_helpers import build_graph, dfs_longest_path
import pandas as pd
import math
from collections import defaultdict

#######try adding the histogram equalization to the videos!!!

######LOAD VIDEO AND MODEL#########
def load_video_as_3d_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

model_location = "/Users/jmanasse/Desktop/cornell/hein_lab/pre_trained_wts/best_yolo_x.pt"
model = YOLO(model_location)
video_location = '/Users/jmanasse/Desktop/cornell/hein_lab/G2Last/2024-11-11TestRound_0043.mov'
video_array = load_video_as_3d_sequence(video_location)

print('Loaded model file from: ' + model_location + ' and video from: ' + video_location)

#####BACKGROUND SUBTRACTION#######
#note: fish who don't move aren't detected
background_image = np.mean(video_array, axis=0)
clean_video = []
for frame in video_array:
  clean_frame = frame-background_image
  _, clean_frame = cv2.threshold(clean_frame, np.mean(clean_frame)-10, 255, cv2.THRESH_BINARY)
  clean_frame = cv2.GaussianBlur(clean_frame, (21,21), 0) #helps to patch belly 'hole'
  clean_frame = clean_frame.astype(np.uint8)
  clean_video.append(clean_frame)
clean_video = np.array(clean_video)


########EQUALIZE VALUES##############
#equalize values in video
equalized_list = []
for frame in video_array:
  gray_picture_array = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray_equalized_image = cv2.equalizeHist(gray_picture_array)
  equalized_image = cv2.cvtColor(gray_equalized_image, cv2.COLOR_GRAY2BGR)
  equalized_list.append(equalized_image)
equalized_array = np.array(equalized_list)
video_array = equalized_array

##########LIGHT DETECTION############
#getting the frames where
epsilon = 0.2
frames_dark = []
frames_light = []
for i in range(len(clean_video)):
  light_box=clean_video[i][130:270, 1130:1270]#box where light shows up
  if np.mean(light_box) > 255-epsilon:
    frames_dark.append(i)
  else:
    frames_light.append(i)

######BEGIN TRACKING HELPERS########

def endpoints_from_skeleton(skeleton,longest_path):
  binaryImage = np.zeros(skeleton.shape)
  for i in range(skeleton.shape[0]):
    for j in range(skeleton.shape[1]):
      if (i,j) in longest_path:
        binaryImage[i][j] = 10
  # Set the end-points kernel:
  h = np.array([[1, 1, 1],
              [1, 10, 1],
              [1, 1, 1]])
  # Convolve the image with the kernel:
  imgFiltered = cv2.filter2D(binaryImage, -1, h)
  plt.imshow(imgFiltered)
  # Extract only the end-points pixels, those with
  # an intensity value of 110:
  endPointsMask = np.where(imgFiltered == 110, 255, 0)
  # The above operation converted the image to 32-bit float,
  # convert back to 8-bit uint
  endPointsMask = endPointsMask.astype(np.uint8)
  x,y = np.where(endPointsMask)
  return binaryImage,x,y

def head_tail_distinguisher(clean_image, longest_path, p1,p2):

  # Find contours using scikit-image
  contours = find_contours(clean_image, level=0.5)
  distances = []

  for y,x in longest_path:
      min_distance = float('inf')
      for contour in contours:
          for c_point in contour:
              cy, cx = c_point
              dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)  # Euclidean distance
              if dist < min_distance:
                  min_distance = dist
      distances.append(min_distance)

  max_idx = np.argmax(distances)
  max_point = longest_path[max_idx]

  dist_1 = np.linalg.norm(np.array(p1) - np.array(max_point))
  dist_2 = np.linalg.norm(np.array(p2) - np.array(max_point))
  nose = None
  if dist_1 < dist_2:
    return p1, p2
  else:
    return p2, p1
  
#speed helper
def fish_speed(head_list):
  #speed is delta x/delta t
  if len(head_list) == 1:
    return 0
  else:
    delta_x = math.dist(head_list[-1],head_list[-2]) #euclidean distance
    delta_t = 1 #wait what's the frame rate again
    return delta_x/delta_t #so speed is in pixels/frame rate ig

#angular velocity helper
'''The absolute angle in radians with respect to the X-axis of the image
(e.g. 0rad indicates the individual is horizontal, looking to the right).
You can convert this to degrees using ANGLE/pi*180.
'''
#ig I can use the skeleton to get the angle?
#it's also just w=v/r or theta*t so there's that
#trex chose the x axis as the axis which the fish rotates around (which x axis? the center of the image or the bottom?
#use the bottom left of the image as the origin for the angle?
def angular_velocity(head_list,speed_list):
  theta = math.atan2(head_list[-1][1], head_list[-1][0]) #angle in radians between x-axis and fish head
  r = math.dist(head_list[-1], (0,0)) #distance from the origin
  omega = speed_list[-1]*math.sin(theta)/r
  return omega

'''
The offset of the tail of an individual to the center line through its body (roughly speaking).
More specifically, it is the y-coordinate of the tail tip,
after normalizing the orientation and position of the individual (as in the posture window).
'''
def midline_offset(tail_list, centroid_list):
  x,y = centroid_list[-1]
  return tail_list[-1][1]-y #i think? I'm not 100% sure this is right

#detect flipping by (1) velocity or (2) a change in head/tail pixel by around the fish length
def track_matching(head_track, tail_track, epsilon=50):
  #rejoin head and tail tracks, epsilon is the swap detection threshold
  for i in range(1,len(head_track)):
    if np.linalg.norm(np.array(head_track[i]) - np.array(head_track[i-1])) > epsilon: #euclidean distance!!
      temp = head_track[i]
      head_track[i] = tail_track[i]
      tail_track[i] = temp
  return head_track, tail_track

#get skeleton and endpoints given a video frame 
def get_structure(frame, box):

  #blur it a bit to cover the belly
  frame = cv2.GaussianBlur(frame, (21,21), 0)
  # frame = cv2.GaussianBlur(frame, (9,9), 0) #try decreasing blur strength
  # Crop the object using the bounding box coordinates
  x1, y1, x2, y2 = box
  ultralytics_crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
  #Fix the contrast
  # ultralytics_crop_object = ((ultralytics_crop_object - ultralytics_crop_object.min()) / (ultralytics_crop_object.max()-ultralytics_crop_object.min())) *255

  # Save the cropped object as an image
  cv2.imwrite('ultralytics_crop.jpg', ultralytics_crop_object)
  #Read back in + threshold
  image = cv2.imread('ultralytics_crop.jpg', 0)
  _, clean_image = cv2.threshold(image, np.mean(image) - 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  #pad image for later
  top, bottom, left, right = 10, 10, 10, 10
  color = [255,255,255]
  clean_image = cv2.copyMakeBorder(clean_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

  #get skeleton
  skeleton = skeletonize(1 - (clean_image / 255))
  skel = np.array(skeleton)
  skel = skel.astype(np.uint8)*255

  graph = build_graph(skeleton)
  start_node = next(iter(graph))
  org_longest_path = dfs_longest_path(graph, start_node)
  path_y, path_x = zip(*org_longest_path)
  path_x = np.array(path_x)
  path_y = np.array(path_y)

#[needs to be debugged]
  # t = np.linspace(0,1,path_x.size)
  # spl_x = UnivariateSpline(t, path_x, s=1)
  # spl_y = UnivariateSpline(t, path_y, s=1)
  # t_smooth = np.linspace(t.min(), t.max(), 20)
  # x_smooth = np.array(spl_x(t_smooth), dtype=int)
  # y_smooth = np.array(spl_y(t_smooth), dtype=int)
  # longest_path = list(zip(y_smooth,x_smooth))

  longest_path = list(zip(path_y,path_x)) 
  binaryImage, x,y = endpoints_from_skeleton(skeleton,longest_path)

#   print(x,y)
  p1=None
  p2=None

  caught_you = False
  try:
    p1 = x[0],y[0]
    p2 = x[-1],y[-1]
  except:
    caught_you = True
  #if euclidean distance between the head and tail isis too small
  #try the non-smoothed version
  if caught_you or np.linalg.norm(np.array(p1) - np.array(p2)) < 20:
    binaryImage, x,y = endpoints_from_skeleton(skeleton,org_longest_path)
    # print(x,y)
    plt.imshow(clean_image)
    p1 = x[0],y[0]
    p2 = x[-1],y[-1]

    # if we only got one point, try filling the hole
    # if caught_you or np.linalg.norm(np.array(p1) - np.array(p2)) < 20:
    if len(x) == 1:
      _, clean_image = cv2.threshold(image, np.mean(image) - 10, 255, cv2.THRESH_BINARY)
      label_im = label(clean_image)
      regions = regionprops(label_im)
      properties = ['area','convex_area','bbox_area', 'extent',
                    'mean_intensity', 'solidity', 'eccentricity',
                    'orientation']
      region_table = pd.DataFrame(regionprops_table(label_im, clean_image,
                  properties=properties))
      print(len(region_table))
      idx = list(region_table['area']).index(min(list(region_table['area'])))
      centroid = (int(regions[idx].centroid[0]), int(regions[idx].centroid[1]))
      clean_image = flood_fill(clean_image,centroid,0)

      skeleton = skeletonize(1 - (clean_image / 255))
      skel = np.array(skeleton)
      skel = skel.astype(np.uint8)*255

      graph = build_graph(skeleton)
      start_node = next(iter(graph))
      org_longest_path = dfs_longest_path(graph, start_node)
      binaryImage, x,y = endpoints_from_skeleton(skeleton,org_longest_path)
    #   print(x,y)
    #   plt.imshow(clean_image)
      p1 = x[0],y[0]
      p2 = x[-1],y[-1]

  #distinguish the head from the tail
  head, tail = head_tail_distinguisher(clean_image, org_longest_path, p1,p2)
  head = (head[1]+x1-10,head[0]+y1-10)
  tail = (tail[1]+x1-10,tail[0]+y1-10)
 
  return binaryImage, head, tail

#get skeleton and endpoints given a video frame [BEST VERSION]
def get_structure1(frame, box):
  x1, y1, x2, y2 = box
  #blur it a bit to cover the belly
  frame = cv2.GaussianBlur(frame, (21,21), 0)
  # frame = cv2.GaussianBlur(frame, (9,9), 0) #try decreasing blur strength
  # Crop the object using the bounding box coordinates
  ultralytics_crop_object = frame[int(y1):int(y2), int(x1):int(x2)]
  #Fix the contrast
  # ultralytics_crop_object = ((ultralytics_crop_object - ultralytics_crop_object.min()) / (ultralytics_crop_object.max()-ultralytics_crop_object.min())) *255

  # Save the cropped object as an image
  cv2.imwrite('ultralytics_crop.jpg', ultralytics_crop_object)
  #Read back in + threshold
  image = cv2.imread('ultralytics_crop.jpg', 0)
  _, clean_image = cv2.threshold(image, np.mean(image) - 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  #pad image for later
  top, bottom, left, right = 10, 10, 10, 10
  color = [255,255,255]
  clean_image = cv2.copyMakeBorder(clean_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

  # if we only got one point, try filling the hole
  # if caught_you or np.linalg.norm(np.array(p1) - np.array(p2)) < 20:

  _, clean_image = cv2.threshold(image, np.mean(image) - 10, 255, cv2.THRESH_BINARY)
  label_im = label(clean_image)
  regions = regionprops(label_im)
  properties = ['area','convex_area','bbox_area', 'extent',
            'mean_intensity', 'solidity', 'eccentricity',
            'orientation']
  region_table = pd.DataFrame(regionprops_table(label_im, clean_image,
            properties=properties))
  if len(region_table) > 1:
    idx = list(region_table['area']).index(min(list(region_table['area'])))
    centroid = (int(regions[idx].centroid[0]), int(regions[idx].centroid[1]))
    clean_image = flood_fill(clean_image,centroid,0)

  #get skeleton
  skeleton = skeletonize(1 - (clean_image / 255))
  skel = np.array(skeleton)
  skel = skel.astype(np.uint8)*255

  graph = build_graph(skeleton)
  start_node = next(iter(graph))
  org_longest_path = dfs_longest_path(graph, start_node)
  path_y, path_x = zip(*org_longest_path)
  path_x = np.array(path_x)
  path_y = np.array(path_y)

  t = np.linspace(0,1,path_x.size)
  spl_x = UnivariateSpline(t, path_x, s=1)
  spl_y = UnivariateSpline(t, path_y, s=1)
  t_smooth = np.linspace(t.min(), t.max(), 20)
  x_smooth = np.array(spl_x(t_smooth), dtype=int)
  y_smooth = np.array(spl_y(t_smooth), dtype=int)
  longest_path = list(zip(y_smooth,x_smooth))

  binaryImage, x,y = endpoints_from_skeleton(skeleton,longest_path)

  print(x,y)
  
  caught_you = False
  try:
    p1 = x[0],y[0]
    p2 = x[-1],y[-1]
  except:
    caught_you = True
  #if euclidean distance between the head and tail isis too small
  #try the non-smoothed version
  if caught_you or np.linalg.norm(np.array(p1) - np.array(p2)) < 20:
    binaryImage, x,y = endpoints_from_skeleton(skeleton,org_longest_path)
    print(x,y)
    plt.imshow(clean_image)
    p1 = x[0],y[0]
    p2 = x[-1],y[-1]

   

  #distinguish the head from the tail
  head, tail = head_tail_distinguisher(clean_image, org_longest_path, p1,p2)
  head = (head[1]+x1-10,head[0]+y1-10)
  tail = (tail[1]+x1-10,tail[0]+y1-10)
  # first_box = clean_image[int(p1[1]-5):int(p1[1]+5), int(p1[0]-5):int(p1[0]+5)]
  # last_box = clean_image[int(p2[1]-5):int(p2[1]+5), int(p2[0]-5):int(p2[0]+5)]
  # head=None
  # tail = None
  # if np.mean(first_box) < np.mean(last_box):
  #     head = (p1[1]+x1-10,p1[0]+y1-10) # back to og coordinates
  #     tail = (p2[1]+x1-10,p2[0]+y1-10)
  # else:
  #     head = (p2[1]+x1-10,p2[0]+y1-10)
  #     tail = (p1[1]+x1-10,p1[0]+y1-10)

  return binaryImage, head, tail
########END TRACKING HELPERS#######

##########FISH DETECTION###########
show_window = True #set this to False if you don't want the window popup


#if the mean value of the clean video frame inside the bounding box < 255
#then keep it
#otherwise discard the tracking id
####NOTE: if the fish is not ID'ed by the model at the 0th frame it will NOT
#be registered in keepsies.
results = model.track(video_array[0], persist=True, iou= 0.1, show_labels=False)
boxes = results[0].boxes.xyxy.cpu()
track_ids = results[0].boxes.id.int().cpu().tolist()
keepsies = []
for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
  x1, y1, x2, y2 = box
  clean_box = clean_video[0][int(y1):int(y2), int(x1):int(x2)]
  if np.mean(clean_box) < 255:
    keepsies.append(track_id)

# model.predictor.trackers[0].reset() #clear track id
super_dict=[]
# for i in range(fish_count):
for i in range(len(keepsies)):
    super_dict.append({'skeleton' : [], 'head': [], 'tail': [], 'centroid': [],
                      'bounding_box' : [], 'angular_velocity' : [], 'speed' : [], 'midline_offset' : []})

head_history = defaultdict(lambda: [])
tail_history = defaultdict(lambda: [])
# for j,frame in enumerate(clean_video):
for j,frame in enumerate(video_array):
  # Get the boxes and track IDs
    results = model.track(frame, persist=True, iou= 0.1,show_labels=False)
    boxes = results[0].boxes.xyxy.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    annotated_frame = results[0].plot()
    # print(track_ids)
    #for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
    i = 0
    for (box, track_id) in zip(boxes, track_ids):
      if track_id not in keepsies:
        continue

      #get bounding box
      x1, y1, x2, y2 = box
      super_dict[i]['bounding_box'].append((x1,y1,x2,y2))
      super_dict[i]['centroid'].append(((x1+x2)/2,(y1+y2)/2))
      #get tail, head, skeleton
      skeleton, head, tail = get_structure(frame, box)
      super_dict[i]['skeleton'].append(skeleton)
      super_dict[i]['head'].append(head)
      super_dict[i]['tail'].append(tail)
      i +=1
      
      heads = head_history[track_id]
      heads.append(head) 
    #   tails = tail_history[track_id]
    #   tails.append(tail) 

      if show_window:
            # Draw the tracking lines
        head_points = np.hstack(heads).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [head_points], isClosed=False, color=(255, 0, 0), thickness=3)

        # tail_points = np.hstack(tails).astype(np.int32).reshape((-1, 1, 2))
        # cv2.polylines(annotated_frame, [tail_points], isClosed=False, color=(0, 0, 255), thickness=3)

        cv2.imshow("YOLO11 Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if show_window:
   cv2.destroyAllWindows()

#perform track matching and tracking variables
for dictionary in super_dict:
  if len(dictionary['head']) == 0:
    continue
  dictionary['head'], dictionary['tail'] = track_matching(dictionary['head'], dictionary['tail'],epsilon=25)
  dictionary['head'] = [(float(x[0]), float(x[1]))for x in dictionary['head']]
  dictionary['tail'] = [(float(x[0]), float(x[1]))for x in dictionary['tail']]
  #get angular velocity, regular velocity, midline offset
  #for some reason this is only saving one point?
  dictionary['speed'].append(0)
  dictionary['angular_velocity'].append(0)
  dictionary['midline_offset'].append(0)
  for i in range(1,len(dictionary['head'])):
    dictionary['speed'].append(fish_speed(dictionary['head'][i-1:i+1]))
    dictionary['angular_velocity'].append(angular_velocity(dictionary['head'][i-1:i+1],dictionary['speed'][i-1:i+1]))
    dictionary['midline_offset'].append(midline_offset(dictionary['tail'][i-1:i+1], dictionary['centroid'][i-1:i+1]))
    
  dictionary['lighting'] = ['dark' if frame in frames_dark else 'light' for frame in range(len(video_array))]


#you might want to save dataframe to excel file like this [it's buggy for some reason. think it might be a 
#package issue]
tracks_sheet_name = 'fish_tracking.xlsx'
with pd.ExcelWriter(tracks_sheet_name, engine='xlsxwriter') as writer:  
    for i,dictionary in enumerate(super_dict):
        track_frame = pd.DataFrame.from_dict(dictionary)
        track_frame.to_excel(writer,
                sheet_name='Fish id '+str(i))
writer.close()
    

print('tracks written to: ' + tracks_sheet_name)
print('All done!')