import cv2;
import os;
import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt;
class SKETCH:
    def __init__(self,archive_path):
        X=[];
        y=[];
        os.chdir(archive_path)
        cat_croquis=[f.path for f in os.scandir(os.getcwd()) if f.is_dir()];
        self.LABEL_NAMES=[os.path.basename(f.path) for f in os.scandir(os.getcwd()) if f.is_dir()];
        for cat in cat_croquis:       
            files_in=[f.path for f in os.scandir(os.path.join(archive_path,cat)) if f.is_file];
            for file_in in files_in:
                originalImage = cv2.imread(file_in)
                grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
                (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)    
                #mise à plat
                X.append(blackAndWhiteImage.ravel());
                y.append(cat_croquis.index(cat));
        self.X=np.asarray(X).reshape(19898,224*224);
        self.y=np.asarray(y);
    def show_examples(self):
        fig,axes=plt.subplots(5,5);
        fig.tight_layout();
        for i in range(5):
            for j in range(5):
                rand=np.random.choice(range(self.X.shape[0]));
                axes[i][j].set_axis_off();
                axes[i][j].imshow(self.__prep_img(rand), cmap="gray");
                axes[i][j].set_title(self.LABEL_NAMES[self.y[rand]]);
        plt.show();
    def __prep_img(self,idx):
        img=self.X[idx].reshape(224,224).astype(np.uint8)/255;
        return img
dataset=SKETCH("/home/bruno.boissie/Travail/artificial_intelligence/uisketch/archive")
print(dataset.X.shape);
dataset.show_examples()