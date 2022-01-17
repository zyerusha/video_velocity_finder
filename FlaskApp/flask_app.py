
from datetime import datetime
import json
import random
import time
import os
from apscheduler.schedulers.background import BackgroundScheduler
from markupsafe import escape
import flask
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_file, send_from_directory, safe_join, abort
from werkzeug.utils import secure_filename
from vel_calc_main import VideoVelCalc
import threading

app = Flask(__name__)
app.uploadVideoName = ""
app.videoVelCalc = VideoVelCalc()

app.config.from_object("config.TestingConfig")

if app.config["ENV"] == "production":
    app.config.from_object("config.ProductionConfig")
elif app.config["ENV"] == "testing":
    app.config.from_object("config.TestingConfig")
else:
    app.config.from_object("config.DevelopmentConfig")


def show_time():
    now = datetime.now().strftime("%A %d, %B %Y  %H:%M:%S")
    return str(f"{escape(now)}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def make_tree(path):
    tree = dict(name=os.path.basename(path), children=[])
    try:
        lst = os.listdir(path)
    except OSError:
        pass  # ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                tree['children'].append(make_tree(fn))
            else:
                tree['children'].append(dict(name=name))
    return tree


def dirtree():
    path = os.path.expanduser(u'~')
    return render_template('dirtree.html', tree=make_tree(path))


@app.route('/processing_done/<filename>')
def processing_done(filename):
    print(filename)
    return render_template("client/processing_done.html", filename=filename)


@app.route("/download/<filename>")
def download(filename):
    try:
        return send_from_directory(app.config["DOWNLOAD_FOLDER"], filename=filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)

@app.route('/static/')
def dirtree():
    path = os.path.join(os.getcwd(), 'static')
    return render_template('client/dirtree.html', tree=make_tree(path))


@app.route('/get_status')
def get_status():
    yolo_progress, vel_progress = app.videoVelCalc.GetProgress()
    progress_str = str(yolo_progress) + ',' + str(vel_progress)
    if(app.videoVelCalc.IsRunning):
        return progress_str
    else:
        return redirect('/')

@app.route('/processing_file/<video_name>', methods=['GET', 'POST'])
def process_file(video_name):
    if request.method == 'GET':
        app.uploadVideoName = video_name
        app.backgroundThread = threading.Thread(target=app.videoVelCalc.GetProgress, args=(), daemon=True)
        app.backgroundThread.start()
        return render_template('client/processing_video_msg.html', filename=video_name)

    if request.method == 'POST':
        full_upload_video_name = safe_join(app.config['UPLOAD_FOLDER'], app.uploadVideoName)

        # if(not app.videoVelCalc.IsRunning):
        app.videoVelCalc = VideoVelCalc()
        app.videoVelCalc.SetCameraParams(30, 15)
        app.videoVelCalc.SetVelCalibarion(2.23694)
        download_video_name = app.videoVelCalc.Run(full_upload_video_name, app.config['DOWNLOAD_FOLDER'], -1, 0)
        filename = os.path.basename(download_video_name)
        return filename
    return render_template('client/processing_video_msg.html', filename=video_name)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            video_name = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)
            full_upload_name = safe_join(app.config['UPLOAD_FOLDER'], video_name)
            file.save(full_upload_name)
            return redirect('/processing_file/'+video_name)

    return render_template("client/file_upload_page.html")


if __name__ == "__main__":
    print(f'ENV is set to: {app.config["ENV"]}')
    app.run(host='0.0.0.0')
