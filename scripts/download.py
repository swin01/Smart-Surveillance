import os
from roboflow import Roboflow

api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)

project = rf.workspace("aswins-workspace").project("appliances-vjnxv-1qloy")
version = project.version(1)

dataset = version.download("yolov7")
