from PIL import Image
import numpy as np
import os

class DataReader():

    def readImages(folderPath, resizeSize, ifBW):
        data = list()

        for file in os.listdir(folderPath):
            img = Image.open(os.path.join(folderPath, file))
            img = img.resize((resizeSize, resizeSize), Image.LANCZOS)

            if ifBW:


            np_img = np.array(img)
            data.append(np_img)
        
        np_data = np.asarray(data)
        return np.asarray(np_data)