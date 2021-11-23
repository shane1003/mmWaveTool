from flask import Flask, render_template, url_for, redirect, request
from werkzeug.wrappers import Request, Response
from werkzeug.datastructures import MultiDict
import json
import os

from Indoor import inputTxGetAll

app = Flask(__name__, template_folder='template')

def imd_parsing(imd):
    scenario_num = imd.getlist('scenario')[0]
    img_name = imd.getlist('images')[0]
    x = imd.getlist('x')[0]
    y = imd.getlist('y')[0]
    tilt = imd.getlist('tilt')[0]
    height = imd.getlist('height')[0]


@app.route("/")
def main():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        print("POSTPOSTPOST")
        return render_template('index.html')

@app.route("/urban", methods=['GET', 'POST'])
def urban():
    imd = request.form
    scenario_num, img_name, x, y, tilt, azimuth = imd_parsing(imd)

    result = os.popen("python grid.py ", id, " ", x, " ", y, " ", azimuth, " ", tilt).read()
    
    return render_template('urban.html')

@app.route("/urban-data", methods=['GET', 'POST'])
def urbanData():
    root_dir = "./urban/resultImage"
    possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.png']
    img_path_list = []
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    img_path = root + '/' + file_name
                    
                    # 경로에서 \를 모두 /로 바꿔줘야함
                    img_path = img_path.replace('\\', '/') # \는 \\로 나타내야함         
                    img_path_list.append(img_path)                         
    print(img_path_list)

    result = json.dumps({'list':img_path_list})
    return result

@app.route("/suburban", methods=['GET', 'POST'])
def suburban():
    os.popen("python ./suburban/LAMS.py")

    return render_template('suburban.html')

@app.route("/suburban-data", methods=['GET', 'POST'])
def suburbanData():
    root_dir = "./suburban/resultImage"
    possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.png']
    img_path_list = []
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    img_path = root + '/' + file_name
                    
                    # 경로에서 \를 모두 /로 바꿔줘야함
                    img_path = img_path.replace('\\', '/') # \는 \\로 나타내야함         
                    img_path_list.append(img_path)                         
    print(img_path_list)

    result = json.dumps({'list':img_path_list})
    return result

@app.route("/indoor", methods=['GET', 'POST'])
def indoor():
    imd = request.form
    scenario_num, img_name, x, y, tilt, height = imd_parsing(imd)
    iT = inputTxGetAll.inputTxGetAllPrediction(x, y, 'resultImage/')
    iT.prediction()
    
    return render_template('indoor.html')

@app.route("/indoor-data", methods=['GET', 'POST'])
def indoorData():
    root_dir = "./indoor/resultImage"
    possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.png']
    img_path_list = []
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in possible_img_extension:
                    img_path = root + '/' + file_name
                    
                    # 경로에서 \를 모두 /로 바꿔줘야함
                    img_path = img_path.replace('\\', '/') # \는 \\로 나타내야함         
                    img_path_list.append(img_path)                         
    print(img_path_list)

    result = json.dumps({'list':img_path_list})
    return result

@app.route("/result", methods=['GET', 'POST'])
def result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port=8888, debug=True)