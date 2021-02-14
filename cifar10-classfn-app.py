import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (32,32)    
        image = ImageOps.fit(image_data, size)
        #image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('cifar10-classfn.hdf5')

st.title("""
          Image Classification based on Cifar-10 dataset
         """
         )

st.write("The dataset on which the model has been trained contains below categories")
st.write("[Aeroplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship]")

file = st.file_uploader("Please upload an image of the relevant category", type=["jpg", "png","JPEG"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    if st.button('Predict'):
        if np.argmax(prediction) == 0:
            st.write("It is an Airplane!")
        elif np.argmax(prediction) == 1:
            st.write("It is an Automobile")
        elif np.argmax(prediction) == 2:
            st.write("It is a Bird")    
        elif np.argmax(prediction) == 3:
            st.write("It is a Cat")    
        elif np.argmax(prediction) == 4:
            st.write("It is a Deer")
        elif np.argmax(prediction) == 5:
            st.write("It is a Dog")
        elif np.argmax(prediction) == 6:
            st.write("It is a Frog")
        elif np.argmax(prediction) == 7:
            st.write("It is a Horse")  
        elif np.argmax(prediction) == 8:
            st.write("It is a Ship")          
        else:
            st.write("It is a Truck")  
    
        
        st.text("Probability (0: Airplane, 1: Automobile, 2: Bird, 3 : Cat, 4 : Deer, 5 : Dog, 6 : Frog, 7 : Horse, 9 : Ship, 10 : Truck)")
        st.write(prediction)
