from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

#from tensorflow.keras import models
from tensorflow import keras
from keras.preprocessing import image
import json
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


img_height, img_width=224,224
with open('./models/imagenet_classes.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)

model_graph = tf.Graph()
with model_graph.as_default():
    tf_session = tf.Session()
    with tf_session.as_default():
        model=keras.models.load_model('./models/MobileNetModelImagenet.h5')




# Create your views here.
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)

    import numpy as np
    predictedLabel=labelInfo[str(np.argmax(predi[0]))]

    context={'filePathName':filePathName,'predictedLabel':predictedLabel[1]}
    return render(request,'index.html',context) 

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 