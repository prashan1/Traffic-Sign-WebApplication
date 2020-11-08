from werkzeug.utils import secure_filename
import numpy as np
from flask import render_template, request , Flask
import tensorflow as tf
import cv2 ,numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model('./Model/weights.h5')

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing vehicle > 3.5 tons' }
            
def preprocessingImage( img_path ):
    img = Image.open( img_path )
    img = np.array( img )
    img = cv2.resize( img , (32,32))
    img = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img)
    img = np.array(img).reshape(1,32,32,1)
    img = img/255.
    
    return img
            
            
@app.route('/')
def home():
    return render_template('index.html')
  
@app.route('/predict' , methods = ['GET','POST'])
def imageUploaded():
    if request.method == 'POST':
        img = request.files['file']
        file_path = secure_filename(img.filename)
        img.save(file_path)
        
        img = preprocessingImage(file_path)
        prediction = model.predict_classes( img )[0]
        imageClass = classes[ prediction ] 
        result = "Predicted Traffic🚦Sign is: " + imageClass
        os.remove(file_path)
        return result

if __name__ == '__main__':
    app.run(debug = True )



















