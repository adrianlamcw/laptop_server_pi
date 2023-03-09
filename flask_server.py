import requests
from flask import Flask, jsonify, send_file
from PIL import Image
from io import BytesIO
import os
import fruit_classifier

app = Flask(__name__)
ip_path = 'http://192.168.121.17'

@app.route('/ping')
def get_data():
    response = requests.get(ip_path + '/ping')
    return jsonify(response.json())

@app.route('/status')
def get_status():
    try:
        response = requests.get(ip_path + '/status')
        return jsonify(response.json())

    except requests.exceptions.RequestException as e:
        # Handle request errors
        return 'Error: {}'.format(str(e))

@app.route('/unlock/<id>')
def get_unlock(id):
    try:
        url = ip_path + f'/unlock/{id}'
        response = requests.get(url)
        if response.status_code == 200:
            print('The request was successful.')
            return 'The request was successful.'

    except requests.exceptions.RequestException as e:
        # Handle request errors
        return 'Error: {}'.format(str(e))

@app.route('/lock/<id>')
def get_lock(id):
    try:
        url = ip_path + f'/lock/{id}'
        response = requests.get(url)
        if response.status_code == 200:
            print('The request was successful.')
            return 'The request was successful.'

    except requests.exceptions.RequestException as e:
        # Handle request errors
        return 'Error: {}'.format(str(e))

@app.route('/camera')
def get_image():
    try:
        response = requests.get(ip_path + '/camera')

        img = Image.open(BytesIO(response.content))
        img_path = os.path.abspath('./Desktop/laptop_server/fridge_images/original.jpeg')
        #img_path = './fridge_images/original.jpeg'
        img.save(img_path)
        
        return send_file(img_path)
    except requests.exceptions.RequestException as e:
        # Handle request errors
        return 'Error: Could not retrieve image: {}'.format(str(e))
    except IOError as e:
        # Handle image processing errors
        return 'Error: Could not process image: {}'.format(str(e))

@app.route('/groceries')
def get_groceries():
    try:
        # response = requests.get(ip_path + '/camera')

        # img = Image.open(BytesIO(response.content))
        # img_path = os.path.abspath('Desktop/fridge_images/original.jpeg')
        # img.save(img_path)

        # RUN MACHINE LEARNING CODE
        food = fruit_classifier.makeBoundingBoxes()
        print(food)
        print(sum(food.values()))
        result = []
        for name in set([f.split()[1] for f in food]):
            obj = {}
            obj['name'] = name
            obj['rotten_count'] = food.get('rotten ' + name, 0)
            obj['fresh_count'] = food.get('fresh ' + name, 0)
            result.append(obj)

        return {
            'total_count': sum(food.values()),
            'data': result
        }

    except requests.exceptions.RequestException as e:
        # Handle request errors
        return 'Error: Could not retrieve image: {}'.format(str(e))
    except IOError as e:
        # Handle image processing errors
        return 'Error: Could not process image: {}'.format(str(e))

if __name__ == '__main__':
    app.run(host='192.168.121.62', port=80, debug=True)