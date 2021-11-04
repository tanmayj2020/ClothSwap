from torch.utils.data import Dataset , DataLoader
import os 
from os import path 
import torch 
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import json 
import numpy as np 

class VitonDataset(Dataset):

    def __init__(self , args):
        super().__init__()
        # Stroing the reference image heigth and width and will be used to resize Cloth , Mask etc
        self.reference_image_height = args.load_height 
        self.reference_image_width = args.load_width 

        # Whether training or testing folder
        self.data_path = path.join(args.dataset_dir ,args.dataset_mode)
        # Person Image directory images
        self.img_names = os.listdir(path.join(self.data_path , "image"))
        

        #Transform for the images ( ToTensor transforms )
        self.transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize((0.5 , 0.5 , 0.5) ,(0.5 , 0.5 , 0.5)) ])
        # Segmentation labels from LIP dataset 
        self.segmentation_labels = {
                0: ['background', [0]],
                1: ['hair', [1, 2]],
                2: ['left_shoe', [18]],
                3: ['noise', [3, 11]],
                4: ['face', [4, 13]],
                5: ['left_arm', [14]],
                6: ['right_arm', [15]],
                7: ['upper', [5, 6, 7]],
                8: ['socks', [8]],
                9: ['bottom', [9, 12]],
                10: ['right_shoe', [19]],
                11: ['left_leg', [16]],
                12: ['right_leg', [17]],
                13: ['neck', [10]],
        }
    def get_parse_agnostic(self , parse , pose_data):
        # Getting Segmentation of Hooman as numpy array 
        parse_array = np.array(parse)
        # Getting upper body as value of 255(White) which will be used a mask later
        parse_upper = (parse_array == 7).astype(np.uint8) * 255
        # Preparing mask for neck  
        parse_neck = (parse_array == 13).astype(np.uint8) * 255
        # Agnostic Image on which other Images will be copied
        agnostic = parse.copy()
        
        # Setting the radius values
        r = 10 

        # Mask arms
        # 5 - Left arm and 2 ,5 , 6, 7 for keft arm poses and 6 for right arm an 5 ,2 ,3 ,4 for right arm pose value 
        for parse_id , pose_id in [(5 ,[ 2 , 5 ,6 ,7]) , (6 ,[5 , 2, 3 ,4 ])]:
            # Creating a new image which wil be used as a mask for the corresponding arm (either left or right)
            img = Image.new("L" , (self.reference_image_width , self.reference_image_height) , 'black')
            img_draw = ImageDraw.Draw(img)
            i_prev = pose_id[0]
            for i in pose_id[1:]:
                # Below line because pose_data[i] = (0 , 0) pose keypoint doesnt exists and was predicted to be zero and hence no line ca be drawn
                if((pose_data[i_prev , 0] == 0 and pose_data[i_prev , 1] == 0) or (pose_data[i , 0] == 0 and pose_data[i , 1] == 0)):
                    continue 
                # Drawing a line between i_prev and current i (Tuple line added for converting the x and y coordinates to tuple respectively)
                img_draw.line([tuple(pose_data[j]) for j in [i_prev , i]], 'white' , r * 10 )
                # Setting the raidus for drawing the ellipse at the keypoints which is Smaller if the last point ie the pojnt of the hand
                radius = r * 4 if i == pose_id[-1] else r *15
                # Getting the ith point x and y coordinate
                point_x , point_y = pose_data[i]
                # White and white used two times one for outline and other for fill 
                img_draw.ellipse([point_x - radius , point_y - radius , point_x + radius , point_y + radius] , 'white' , 'white')
                i_prev = i 
            # Defining the final mask for the arm from both pose data and parse data and preserving the hand 
            arm = np.array(img) * (parse_array == pose_id)
            # 0 is the fill value ie whereever the mask is we are filling it with the background value and None for starting position of 0 to be the (0 , 0) of the original image
            # Wherever we have the mask value of 256 at that pixel 0 value will be filled in orignal image and everywhere else original pixel values will be retained
            agnostic.paste(0 , None , Image.fromarray(np.uint8(arm) , 'L'))
        
        # Removing the upper and the neck information 
        agnostic.paste( 0 , None , Image.fromarray(parse_upper , 'L'))
        agnostic.paste( 0 , None , Image.fromarray(parse_neck , 'L'))


        # Returning the segmentation mask
        return agnostic

    def get_image_agnostic(self, img,parse, pose_data):
        # Numpyising the parse array
        parse_array = np.array(parse)
        # Getting the parse head (Face mask)
        parse_head = (parse_array == 4).astype("np.uint8") * 255
        # Getting the parse lower mask
        parse_lower = ((parse_array == 9).astype(np.uint8) + (parse_array == 11).astype(np.uint8) + (parse_array == 12) +  (parse_array == 2).astype(np.uint8) +
                       (parse_array == 10).astype(np.uint8)) * 255
        # Copying the person Image to agnostic for later use of masking 
        agnostic = img.copy()
        # Agnostic draw
        agnostic_draw= ImageDraw.Draw(agnostic)

        # Defining the radius 
        r = 20 
        # Scaling the hip points so that they come closer to each other 
        # Length between the shoulders
        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        # Legth between the hip pose keypoints 
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])

        # Getting the mid point of the hip joint 
        mid_point =(pose_data[12] + pose_data[9])/2

        # Scaling the points
        # Pose Data[9] getting closer to the midpoint 
        pose_data[9] = mid_point + (pose_data[9] - mid_point) / length_a * length_b
        # POse Data[12] getting closer to the mid point 
        pose_data[12] = mid_point + (pose_data[12] - mid_point) / length_a * length_b

        # Drawing a line between shoulder keypoints
        agnostic_draw.line([tuple(pose_data[j]) for j in [2 , 5]] , 'gray' , width=r *10)
        # Drawing ellipses at the shoulder keypoints ie at 2 and 5 of specified radius 
        for i in [2 , 5]:
            pointx , pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        # Drawing the line between the hand keypoints wherein i_prev corresponds to i - 1 
        for i in [3 , 4, 6 ,7 ]:
            if((pose_data[i - 1 , 0] == 0 and pose_data[i-1 , 1] == 0) or (pose_data[i , 0] == 0 and pose_data[i , 1] == 0)):
                continue 
            agnostic_draw.line([tuple(pose_data[j]) for j in [i-1 , i]] , 'gray' , width = r * 10 )
            pointx , pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        
        # Making ellipses on the pose keypoints 9 and 12 
        for i in [9 ,12]:
            pointx ,pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        # Left shoulder and left hip
        agnostic_draw.line([tuple(pose_data[i]) for i in [2,9]] , 'gray' , width =r*6)
        # Right shoulder and right hp 
        agnostic_draw.line([tuple(pose_data[j]) for j in [5 , 12]] , 'gray' , width=r*6)
        # Hip keypoints
        agnostic_draw.line([tuple(pose_data[i]) for i in [9 ,12]] , 'gray' , width = r* 12)
        # Drawing a polygon after drawing the line 
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2 ,5 , 12 , 9]] , 'gray' , 'gray')
        # mask the neck 
        pointx , pointy = pose_data[1]
        # Drawing the rectangle arounf the neck
        agnostic_draw.rectangle((pointx - r * 7 , pointy - r*7 , pointx + r*7 ,pointy + r*7) , 'gray' , 'gray')
        
        
        # Pasting the upper head mask on the image 
        agnostic.paste(img , None , Image.fromarray(parse_head , 'L'))
        # Pasting hte lowe body mask onto the image
        agnostic.paste(img , None , Image.fromarray(parse_lower , "L"))

        return agnostic

        
        





    def __getitem__(self, index):

        # Person Image
        image_name = self.img_names[index]

        # Target Clothing Image (For training it will be the reference persons cloth)
        cloth_image = Image.open(path.join(self.data_path , "cloth" , image_name))
        # Resizing target clothing image according to the reference person Image  
        cloth_image = TF.resize(cloth_image , (self.reference_image_height , self.reference_image_width) , interpolation=InterpolationMode.BILINEAR)
        # Normalizing clothes in range [-1 ,1]
        cloth_image = self.transform(cloth_image)
        

        # Reference Image extension 
        ext = image_name.split(".")[-1]


        # CLoth Mask code
        # Replacing the clothing mask name
        cmask_name = image_name.replace(f".{ext}" ,".png")
        # Opening the clothing mask
        cloth_mask = Image.open(path.join(self.data_path , "cloth-mask" , cmask_name)).convert("L")
        # Resizing the clothing mask 
        cloth_mask = TF.resize(cloth_mask , (self.reference_image_height , self.reference_image_width) , interpolation=InterpolationMode.NEAREST)
        # Converting the mask to numpy array
        cloth_mask = np.array(cloth_mask)
        # Making pixel values greator than 254 equal to 1 and rest other pixel values to 0 
        cloth_mask = (cloth_mask >= 128).astype(np.float32)
        # Converting the pixel values from numpy array to pytorch tensor 
        cloth_mask = torch.from_numpy(cloth_mask) # [0, 1]
        # Adding a new dimenaion to the clothing mask 
        cloth_mask.unsqueeze_(0)

        # Pose RGB Image 
        #Replacing the reference image name by corresponding pose image name
        pose_name = image_name.replace(f".{ext}" , "_rendered.png")
        # Opening the pose RBG image
        pose_rgb = Image.open(path.join(self.data_path ,"openpose-img" ,pose_name)).convert("RGB")
        # Resizing and Normalising persons RBG pose
        pose_rgb = TF.resize(pose_rgb , (self.reference_image_height , self.reference_image_width) , Interpolation = InterpolationMode.BILINEAR)
        pose_rgb = self.transform(pose_rgb) #[-1 , 1]


        # Pose JSON keypoints 
        #Getting pose name
        pose_name = image_name.replace(f".{ext}" , "_keypoints.json")
        # Defining the complete pose path 
        pose_path = path.join(self.data_path , "openpose-json" , pose_name)
        #Opening the pose
        with open(pose_path) as f:
            # Reading the pose
            pose_label = json.load(f)
            # Getting the pose data 
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            # Converting pose data to numpy array
            pose_data = np.array(pose_data)
            # Getting the (X , Y) coordinated of 25 keypoints and removing their confidence score
            pose_data = pose_data.reshape((-1 , 3))[: , :2]


        # Parse Image (Segmentation Map of Reference Person )
        parse_name = image_name.replace(f".{ext}" , ".png")
        parse = Image.open(path.join(self.data_path , "image-parse" , parse_name))
        parse = np.array(parse)
        # Using the parse labels from parse dataset to set the segmentation mask
        # Below  code to avoid different labels for same mask Face (4 , 13) to change to 4
        parse_orig = parse.copy()
        for key , value in self.segmentation_labels.items():
            # Value is list of labels for each body part 
            for label in value[1]:
                if label != key:
                    # Changing parse where different label
                    parse[parse_orig == label] = key
        # deleting parse original as it is of no use now
        del parse_orig

        # Converting parse back to Image 
        parse = Image.fromarray(parse)
        # Resizing the parsing image to downsample it so size becomes (256 , 192) for training the GMM and Segmentation generator
        parse_down = TF.resize(parse ,(256 , 192) , interpolation=InterpolationMode.NEAREST)
        # Converting to pytorch tensor from numpy array first increasing the segmentation mask along dimensiom 1 and then conevrting to tensor and uint8 
        parse_down = torch.from_numpy(np.array(parse_down)[None]).type(torch.uint8)

        # Resizing the human segementation mask to original reference image 
        parse = TF.resize(parse , (self.reference_image_height, self.reference_image_width) ,interpolation=InterpolationMode.NEAREST )
        # Cloth agnostic segementation
        parse_agnostic = self.get_parse_agnostic(parse , pose_data)
        
        #Converting parse agnostic to a tensor with a single channel from PIL Image 
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).type(torch.uint8)

        # Loading the person image
        person_image= Image.open(path.join(self.data_path , "image" , image_name))
        # Resizing the Image
        person_image = TF.resize(person_image , (self.reference_image_height , self.reference_image_width), interpolation=InterpolationMode.BILINEAR)
        # Getting the clothing agnostic Image
        img_agnostic = self.get_image_agnostic(person_image , parse , pose_data)
        # Transforming the Image agnostic between [-1 , 1] and a tenor
        img_agnostic = self.transform(img_agnostic) 
        # Transforming the Image array between [-1 , 1]      
        img = self.transform(person_image)
        # Creating the result dictionary 
        result = {
            'img': img,
            'img_agnostic': img_agnostic,
            'parse_target_down': parse_down,
            'parse_agnostic': parse_agnostic,
            'pose': pose_rgb,
            'cloth': cloth_image,
            'cloth_mask': cloth_mask,
        }
        # Returning the result 
        return result

    def __len__(self):
        #Returning the overall length of the array 
        return len(self.img_names)



# DataLoader class
class VITONDataLoader(DataLoader):
    def __init__(self , args , dataset):
        # Calling the parent dataloader class
        super().__init__()
        # Assining the dataset variable to dataset
        self.dataset = dataset 
        



