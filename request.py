# request.py
# !python3 request.py /path/2/ur/img
import requests
from argparse import ArgumentParser
import os
import cv2
import numpy as np
from io import BufferedReader, BytesIO
import base64
import threading, queue
import multiprocessing as mp
import functools

from utils import WebcamVideoStream, FPS

EVERY_N_FRAME = 4
URL = 'http://0.0.0.0:5000/predict'

def img_2_io_wrap_img(img):
    retval, jpg_enc_img = cv2.imencode(".jpg", img)
    img_bytes = jpg_enc_img.tobytes()		#将array转化为二进制类型
    io_bytes = BytesIO(img_bytes)		#转化为_io.BytesIO类型
    # io_bytes.name = "Cat03.jpg"		#名称赋值
    return BufferedReader(io_bytes)		#转化为_io.BufferedReader类型

def base64_encode(img):
    retval, jpg_enc_img = cv2.imencode(".jpg", img)
    return base64.b64encode(jpg_enc_img)

def single_image_mode(args):
    assert os.path.exists(args.image_path) 
    resp = requests.post(URL, files={"file": open(args.image_path,'rb')})  

    assert resp.status_code == 200, f"request failed, status code: {resp.status_code}"
    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = resp.json()
    cv2.putText(img, f"V:{result['velance']:.2f}, A:{result['arousal']:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow(img)
    cv2.waitkey(0)

def real_time_mode(args):
    cap = cv2.VideoCapture(0)
    count = 0
    while(cap.isOpened()):
        if count%5 == 0:
            count = 0
            fps = FPS().start() 
        elif (count+1)%5 == 0:
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.1f}".format(fps.fps()))
        fps.update()
        count += 1

        ret, frame = cap.read()
        if ret: 
            # resp = requests.post(URL, data=base64_encode(frame))
            resp = requests.post(URL, files={"image_io": img_2_io_wrap_img(frame)})
            assert resp.status_code == 200, f"request failed, status code: {resp.status_code}"
            result = resp.json()
            cv2.putText(frame, f"V:{result['velance']:.2f}, A:{result['arousal']:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class SingleThreadClass:
    def __init__(self, id):
        self.id = id
        self.cap = cv2.VideoCapture(id)

    def main(self):
        count = 0
        while True:
            if count%5 == 0:
                count = 0
                fps = FPS().start() 
            elif (count+1)%5 == 0:
                fps.stop()
                print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                print("[INFO] approx. FPS: {:.1f}".format(fps.fps()))
            fps.update()
            count += 1

            ret, frame = self.cap.read()
            resp = requests.post(URL, data=base64_encode(frame))
            assert resp.status_code == 200, f"request failed, status code: {resp.status_code}"
            result = resp.json()
            cv2.putText(frame, f"V:{result['velance']:.2f}, A:{result['arousal']:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("Video", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        


def frame_reader(queue_from_cam=None):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    queue_from_cam.put(frame)

def poster(queue_from_cam, queue_for_out):
    frame = queue_from_cam.get()
    resp = requests.post(URL, data=base64_encode(frame))
    assert resp.status_code == 200, f"request failed, status code: {resp.status_code}"
    result = resp.json()
    cv2.putText(frame, f"V:{result['velance']:.2f}, A:{result['arousal']:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    queue_for_out.put(frame)

def smap(f, kwargs):
    print(kwargs)
    return f(**kwargs)

class MultiThreadingDetection:
    def __init__(self, id): 
        self.cap = cv2.VideoCapture(id)
        # self.cap = WebcamVideoStream(src=id).start()
        
    def main(self):
        q = queue.Queue()
        count = 0
        while True:
            if count%5 == 0:
                count = 0
                fps = FPS().start() 
            elif (count+1)%5 == 0:
                fps.stop()
                print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                print("[INFO] approx. FPS: {:.1f}".format(fps.fps()))
            fps.update()
            count += 1

            cam = threading.Thread(target=frame_reader, args=(q,))
            cam.start()
            cam.join()
            frame = q.get()
            # q.task_done()
            resp = requests.post(URL, data=base64_encode(frame))
            assert resp.status_code == 200, f"request failed, status code: {resp.status_code}"
            result = resp.json()
            cv2.putText(frame, f"V:{result['velance']:.2f}, A:{result['arousal']:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("Video", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        

class MultiProcessingDetection:
    def __init__(self, id):
        self.id = id
        m = mp.Manager()
        self.queue_from_cam = m.Queue()
        self.queue_for_out= m.Queue()
        self.cap = cv2.VideoCapture(0)

    def main(self):
        count = 0
        while True:
            if count%5 == 0:
                count = 0
                fps = FPS().start() 
            elif (count+1)%5 == 0:
                fps.stop()
                print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
                print("[INFO] approx. FPS: {:.1f}".format(fps.fps()))
            fps.update()
            count += 1

            ret, frame = self.cap.read()
            
            self.queue_from_cam.put(frame)
            with mp.Pool(2) as pool:
                pool.starmap(poster, [(self.queue_from_cam, self.queue_for_out)])

            frame = self.queue_for_out.get()
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        

if __name__ == '__main__':
    arg_parser = ArgumentParser(
    usage='Usage: python ' + __file__ + '[-p <image_path>] [--help]\nIf image_path is not passed then it will run in real time mode'
    )
    arg_parser.add_argument('-p', '--image-path', type=str, help="path of image to predict if real-time mode is not used")


    args = arg_parser.parse_args()

    if args.image_path:
        single_image_mode(args)
    else:
        print("\nreal_time_mode\n")
        real_time_mode(args)

        # print("\nsingle thread\n")
        # SingleThreadClass(0).main()

        # print("\nMultiThread\n")
        # MultiThreadingDetection(0).main()

        # print("\nMultiProcess\n")
        # MultiProcessingDetection(0).main()
    cv2.destroyAllWindows()    
