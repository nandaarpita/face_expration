'''from flask import Flask,render_tempate,request
app=Flask(__name__)
@app.route('/upload',methods=['POST','GET'])
def upload():
	if request.method=='POST'
	return render_template("email.html")

if __name__ == '__main__':
   app.run(debug = True)
'''


from flask import Flask, render_template, request,jsonify, send_file
from werkzeug.utils  import secure_filename
import glob
import dynamic_analysis as dl_alis
import sqlite3 as sql
import requests
import json
import urllib.request
import os
import subprocess
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

    


@app.route('/upload')
def upload():
   return render_template('upload.html')
	

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'


@app.route('/get_data_list', methods = ['GET', 'POST'])
def get_data_list():
   data = glob.glob('*.webm')
   return jsonify({'data':data})


# @app.route("/post/<url>")
# def post(url):
#     url="http://ec2-13-233-163-187.ap-south-1.compute.amazonaws.com:8080/TalentPool/api/v1/getVideoURL"  
#     the_post = get_from_url(introVideoDownloadUrl)
#     response = make_response_from_entity(the_post)
#     return response   
      

@app.route('/get_data_analyze', methods = ['POST'])
def get_data_analyze():
   try:
      data = request.json
      video_name=downlink(data['introVideoDownloadUrl'])
      print(data)
      #analysis = dl_alis.analyze(data['introVideoDownloadUrl'],data['candidateUniqueId'])
      analysis = dl_alis.analyze(video_name,data['candidateUniqueId'])

      analysis['candidateUniqueId'] = data['candidateUniqueId']
      analysis['status'] = True
      os.remove(video_name)
      print("video file deleted")
      #return send_file(data, mimetype='image/png')
      return jsonify(analysis)
   except Exception as e:
      print(e)
      return jsonify({'status':False})



def downlink(url):
   name = url.split('/')[-1]
   try:
      print("Downloading starts...\n")
      urllib.request.urlretrieve(url, name)
      print("Download completed..!!")
      return name     
   except Exception as e:
      print(e)
'''

process= subprocess.Popen( ('ls', '-l', '/tmp'), stdout=subprocess.PIPE)

for line in process.stdout:
        pass

subprocess.call( ('ps', '-l') )
process.wait()
print ("after wait")
subprocess.call( ('ps', '-l') )
'''

if __name__ == '__main__':
   app.run(host='0.0.0.0',port=5002,debug = True)
