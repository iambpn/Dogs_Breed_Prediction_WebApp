from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from collections import OrderedDict
import json
from PIL import Image
import numpy as np
from tensorflow import keras

app = Flask(__name__)
max_size = 10 * 1024 * 1024  # LIMIT : MAX UPLOAD SIZE TO 10 MB
breed_list = [
    'Afghan_hound',
    'African_hunting_dog',
    'Airedale',
    'American_Staffordshire_terrier',
    'Appenzeller',
    'Australian_terrier',
    'Bedlington_terrier',
    'Bernese_mountain_dog',
    'Blenheim_spaniel',
    'Border_collie',
    'Border_terrier',
    'Boston_bull',
    'Bouvier_des_Flandres',
    'Brabancon_griffon',
    'Brittany_spaniel',
    'Cardigan',
    'Chesapeake_Bay_retriever',
    'Chihuahua',
    'Dandie_Dinmont',
    'Doberman',
    'English_foxhound',
    'English_setter',
    'English_springer',
    'EntleBucher',
    'Eskimo_dog',
    'French_bulldog',
    'German_shepherd',
    'German_short-haired_pointer',
    'Gordon_setter',
    'Great_Dane',
    'Great_Pyrenees',
    'Greater_Swiss_Mountain_dog',
    'Ibizan_hound',
    'Irish_setter',
    'Irish_terrier',
    'Irish_water_spaniel',
    'Irish_wolfhound',
    'Italian_greyhound',
    'Japanese_spaniel',
    'Kerry_blue_terrier',
    'Labrador_retriever',
    'Lakeland_terrier',
    'Leonberg',
    'Lhasa',
    'Maltese_dog',
    'Mexican_hairless',
    'Newfoundland',
    'Norfolk_terrier',
    'Norwegian_elkhound',
    'Norwich_terrier',
    'Old_English_sheepdog',
    'Pekinese',
    'Pembroke',
    'Pomeranian',
    'Rhodesian_ridgeback',
    'Rottweiler',
    'Saint_Bernard',
    'Saluki',
    'Samoyed',
    'Scotch_terrier',
    'Scottish_deerhound',
    'Sealyham_terrier',
    'Shetland_sheepdog',
    'Shih-Tzu',
    'Siberian_husky',
    'Staffordshire_bullterrier',
    'Sussex_spaniel',
    'Tibetan_mastiff',
    'Tibetan_terrier',
    'Walker_hound',
    'Weimaraner',
    'Welsh_springer_spaniel',
    'West_Highland_white_terrier',
    'Yorkshire_terrier',
    'affenpinscher',
    'basenji',
    'basset',
    'beagle',
    'black-and-tan_coonhound',
    'bloodhound',
    'bluetick',
    'borzoi',
    'boxer',
    'briard',
    'bull_mastiff',
    'cairn',
    'chow',
    'clumber',
    'cocker_spaniel',
    'collie',
    'curly-coated_retriever',
    'dhole',
    'dingo',
    'flat-coated_retriever',
    'giant_schnauzer',
    'golden_retriever',
    'groenendael',
    'keeshond',
    'kelpie',
    'komondor',
    'kuvasz',
    'malamute',
    'malinois',
    'miniature_pinscher',
    'miniature_poodle',
    'miniature_schnauzer',
    'otterhound',
    'papillon',
    'pug',
    'redbone',
    'schipperke',
    'silky_terrier',
    'soft-coated_wheaten_terrier',
    'standard_poodle',
    'standard_schnauzer',
    'toy_poodle',
    'toy_terrier',
    'vizsla',
    'whippet',
    'wire-haired_fox_terrier'
]
model = None

def getModel():
    load_model = keras.models.load_model('./static/model/mobilenet_finetuned_80.h5')
    print('Keras Model Loading finished.')
    return load_model

model = getModel()

def preprocessImage(image,target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = keras.preprocessing.image.img_to_array(image)
    image = keras.applications.mobilenet.preprocess_input(image)
    image = np.expand_dims(image,0)
    return image


@app.route('/')
def index():
    return render_template('Prediction/index.html')


@app.route('/about')
def about():
    return render_template('Prediction/about.html',breed_list = breed_list,length=int(len(breed_list)/3))


supported_type = ['image/png', 'image/jpeg']


@app.route('/prediction', methods=['POST'])
def prediction():
    if 'file' not in request.files:
        # No file
        return {"Error": "No File Object Found."}, 400

    if request.content_length > (max_size+1000):
        return {"Error": f"File size must be less than {max_size/(1024*1024)} MB"},400

    file = request.files['file']

    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return {"Error": "File not selected. Empty File!"}, 400

    mime_type = file.content_type
    if file and mime_type in supported_type:
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # to save the Image
        image = Image.open(file)
        processed_image = preprocessImage(image,target_size=(224,224))
        global model
        if model == None:
        	model = getModel()
        predicted_data = model.predict(processed_image)
        # class_index = np.argmax(predicted_data)
        sorted_index = predicted_data.argsort()
        top_5_prediction = OrderedDict()
        for i in range(1,6):
            top_5_prediction[breed_list[sorted_index[0,-i]]] = str(predicted_data[0,sorted_index[0,-i]])

        # success return
        return json.dumps(top_5_prediction), 200
    else:
        return {"Error": "File type not supported."}, 400



@app.errorhandler(404)
def exception(ex):
    return render_template('404.html')

if __name__ == '__main__':
    print('Loading Keras Model.')
    app.run()
