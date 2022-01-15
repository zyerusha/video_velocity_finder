
from datetime import datetime
import json
import time
import os
from apscheduler.schedulers.background import BackgroundScheduler
from markupsafe import escape
import flask
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_file, send_from_directory, safe_join, abort
from werkzeug.utils import secure_filename
from shutil import copy2
from celery import Celery


def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


app = Flask(__name__)
app.config.from_object("config.TestingConfig")

if app.config["ENV"] == "production":
    app.config.from_object("config.ProductionConfig")
elif app.config["ENV"] == "testing":
    app.config.from_object("config.TestingConfig")
else:
    app.config.from_object("config.DevelopmentConfig")

app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)

celery = make_celery(app)


@celery.task()
def add_together(a, b):
    return a + b


def show_time():
    now = datetime.now().strftime("%A %d, %B %Y  %H:%M:%S")
    return str(f"{escape(now)}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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


@app.route('/processing_file/<video_name>')
def process_file(video_name):
    full_upload_video_name = safe_join(app.config['UPLOAD_FOLDER'], video_name)
    output_file_video_name = "out_" + video_name
    full_download_video_name = safe_join(app.config['DOWNLOAD_FOLDER'], output_file_video_name)


    show_time()
    time.sleep(10)
    show_time()

    copy2(full_upload_video_name, full_download_video_name)

    # data = render_template('processing_video_msg.html')
    # result = add_together.delay(23, 42)
    # result.wait()  # 65
    # print(f"result: :{result}")
    return redirect('/processing_done/' + output_file_video_name)


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
