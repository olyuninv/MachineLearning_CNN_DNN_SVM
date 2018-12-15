from PIL import Image
import numpy as np
import os

class DataReader():

    def readImages(folderPath, label, resizeSize, ifBW):
        data = list()

        min_size_image = 200
                
        for file in os.listdir(folderPath):
            img = Image.open(os.path.join(folderPath, file))
            img = img.resize((resizeSize, resizeSize), Image.LANCZOS)

            if ifBW:
                img = img.convert('L')
            
            np_img = np.array(img)
            
            if np_img.shape[0] < min_size_image: 
                min_size_image = np_img.shape[0]
            
            if np_img.shape[1] < min_size_image: 
                min_size_image = np_img.shape[1]

            data.append((np_img, label))
        
        print ("Min image dimension size: %i" % min_size_image)
        np_data = np.asarray(data)
        return np_data