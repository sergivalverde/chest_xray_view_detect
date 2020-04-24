# --------------------------------------------------
# Detect chest view using chest x-ray
#
# Sergi Valverde 2020
# svalverde@eia.udg.edu
#
# WARNING: this is a research tool, do not use it for diagnostic
#          without a clinical validation
# --------------------------------------------------

import os
import argparse
import time
import docker
from pyfiglet import Figlet

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":

    # Parse input options
    parser = argparse.ArgumentParser(
        description=" Chest x-ray view inference (VICOROB)")
    parser.add_argument('--input_image',
                        action='store',
                        help='Input image to process (mandatory)')
    parser.add_argument('--model',
                        default='vicorob',
                        help='Model to use for inference (not implemented)')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use GPU for computing (default=false)')
    parser.add_argument('--update',
                        action='store_true',
                        help='Update the Docker image')

    opt = parser.parse_args()
    GPU_USE = opt.gpu
    UPDATE = opt.update

    # --------------------------------------------------
    # Docker image
    # - update docker image at init
    # --------------------------------------------------

    client = docker.from_env()
    CONTAINER_IMAGE = 'sergivalverde/docker-covid-chest:latest'

    if UPDATE:
        print('Updating the Docker image')
        client.images.pull(CONTAINER_IMAGE)

    # --------------------------------------------------
    # SET PATHS
    # Convert input path into an absolute path
    #
    # DATA_FOLDER: abs path where the input scan lives
    # IMAGE_PATH: abs path to the input scan
    # --------------------------------------------------
    input_image = opt.input_image
    if str.find(input_image, '/') >= 0:
        if os.path.isabs(input_image):
            (im_path, im_name) = os.path.split(input_image)
            DATA_PATH = im_path
            IMAGE_PATH = im_name
        else:
            (im_path, im_name) = os.path.split(input_image)
            DATA_PATH = os.path.join(os.getcwd(), im_path)
            IMAGE_PATH = im_name
    else:
        DATA_PATH = os.getcwd()
        IMAGE_PATH = opt.input_image

    # --------------------------------------------------
    # Docker options
    # - docker container paths
    # - volumes to mount
    # - command
    # - runtime
    # --------------------------------------------------

    # docker user
    UID = str(os.getuid())
    DOCKER_USER = UID + ':1000'

    # docker container paths
    DOCKER_DATA_PATH = '/home/docker/data'

    # volumes to mount
    VOLUMES = {DATA_PATH: {'bind': DOCKER_DATA_PATH, 'mode': 'rw'}}
    # MODEL_PATH: {'bind': DOCKER_MODEL_PATH, 'mode': 'rw'}}

    # Internal python command
    COMMAND = 'python /home/docker/src/inference/predict_view.py' + \
              ' --input_scan ' + os.path.join(DOCKER_DATA_PATH, IMAGE_PATH) + \
              ' --output_path ' + DOCKER_DATA_PATH

    # --------------------------------------------------
    # run the container
    # --------------------------------------------------

    t = time.time()

    f = Figlet(font="slant")
    print("--------------------------------------------------")
    print(f.renderText("Chest x-ray view"))
    print("Detect chest x-ray view")
    print(" ")
    print("Universitat de Girona")
    print("(c) Computer Vision and Robotics Institute (VICOROB)")
    print("https://vicorob.udg.edu")
    print(" ")
    print("version: v0.1 (Docker)")
    print("--------------------------------------------------")
    print(" ")
    print("Image information:")
    print("input path:", DATA_PATH)
    print("input image:", IMAGE_PATH)
    print("Using GPU:", GPU_USE)
    print(" ")

    if GPU_USE:
        client.containers.run(image=CONTAINER_IMAGE,
                              command=COMMAND,
                              user=DOCKER_USER,
                              runtime='nvidia',
                              remove=True,
                              volumes=VOLUMES)
    else:
        client.containers.run(image=CONTAINER_IMAGE,
                              user=DOCKER_USER,
                              command=COMMAND,
                              remove=True,
                              volumes=VOLUMES)

    # update probability from the container
    tmp_prob_path = os.path.join(DATA_PATH, '.view.txt')

    if os.path.exists(tmp_prob_path):
        f = open(tmp_prob_path, 'r')
        prob = f.read()
        # remove tmp file
        os.remove(tmp_prob_path)
    else:
        prob = "?"

    print('Current view:', prob)
    print("--------------------------------------------------")
    print('Computing time: %0.2f sec.' % (time.time() - t))
