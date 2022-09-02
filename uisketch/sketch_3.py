import gc
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
        tab=np.genfromtxt("labels.csv",delimiter=",",dtype="U75",skip_header=1)
        #print(tab.shape)
        self.LABEL_NAMES=[os.path.basename(f.path) for f in os.scandir(os.getcwd()) if f.is_dir()];
        liste=tab[:,0]
        np.random.shuffle(liste);
        for file_in in liste:
                originalImage = cv2.imread(os.path.join(archive_path,file_in))
                grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
                (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)    
                #mise à plat
                X.append(blackAndWhiteImage.ravel());
                y.append(file_in.split('/')[0]);
        self.X=np.asarray(X).reshape(19000,224*224);
        self.y=np.asarray(y);
    def show_examples(self):
        fig,axes=plt.subplots(5,5);
        fig.tight_layout();
        for i in range(5):
            for j in range(5):
                rand=np.random.choice(range(self.X.shape[0]));
                axes[i][j].set_axis_off();
                axes[i][j].imshow(self.__prep_img(rand), cmap="gray");
                axes[i][j].set_title(self.y[rand]);
        plt.show();
    def __prep_img(self,idx):
        img=self.X[idx].reshape(224,224).astype(np.uint8)/255;
        return img
    def train_test_split(self):
        #Ensemble d'entrainement (5/6eme) du total.
        X_train=self.X[:15833];
        y_train=self.y[:15833];
        #Ensemble de test le reste des lignes
        X_test=self.X[15833:];
        y_test=self.y[15833:];
        return X_train,y_train,X_test,y_test
    def all_data(self):
        return self.X,self.y
class NearestNeighbor:
    def __init__(self,distance_func='l1'):
        self.distance_func=distance_func;
    def train(self,X,y):
        #X est une matrice N x D chaque ligne est un exemple de test y est une matrice N x 1 de valeurs correctes
        self.X_tr=X.astype(np.float32)
        self.y_tr=y
    def predict(self,X):
        #X est une matrice M x D dans laquelle chaque ligne est un exemple de test 
        X_te=X.astype(np.float32)
        num_test_examples=X.shape[0]
        y_pred=np.zeros(num_test_examples,self.y_tr.dtype)
        for i in range(num_test_examples):
            print("I="+str(i))
            if self.distance_func=='l2':
                distances=np.sum(np.sqare(self.X_tr-X_te[i]),axis=1)
            else:
                distances=np.sum(np.abs(self.X_tr-X_te[i]),axis=1)
            smallest_dist_idx=np.argmin(distances)        
            del distances
            gc.collect()
            y_pred[i]=self.y_tr[smallest_dist_idx]
dataset=SKETCH("/home/bruno.boissie/Travail/artificial_intelligence/uisketch/archive")
X_train,y_train,X_test,y_test=dataset.train_test_split()
X,y=dataset.all_data();
#print(dataset.X.shape);
#dataset.show_examples()
nn=NearestNeighbor();
nn.train(X_train,y_train);
print("Prediction")
y_pred=nn.predict(X_test[:10]);
print("Precision")
accuracy=np.mean(y_test[:10]==y_pred)
print(accuracy)