from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import cv2
import numpy as np
import argparse
import argparse
import os
from pathlib import Path
import sys

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from pymongo import MongoClient
from datetime import datetime, timedelta
import uuid  # For generating unique IDs
import requests
# from datetime import datetime
# Define the endpoint URL
url = "http://localhost:5000/"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]
    
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 
    bbox = face_detector([img])[0]
    # print(bbox.shape)
    bounding_boxes = []

    for face in bbox:
        bbx = face.flatten()[:4].astype(int)
        pred = anti_spoof([increased_crop(img, bbx, bbox_inc=1.5)])[0]
        score = pred[0][0]
        label = np.argmax(pred)   
        # print(bbx, score, label)
        bounding_boxes.append((bbx, score, label))
    # print(bounding_boxes)
    return bounding_boxes
    # return bounding_boxes, label, score

def get_database():
    # Create a connection to the local MongoDB instance
    client = MongoClient("localhost", 27017)
    # Return the desired database
    return client['employee']

def check_last_entry(person_name, attendance_collection):
    # Retrieve the most recent entry for the specified person_name
    last_entry = attendance_collection.find_one({"name": person_name}, sort=[("_id", -1)])
    # print("Last entry -->", last_entry)
    if last_entry:
        # Extract the date string from the last entry
        last_date_str = last_entry["date"]
        # Parse the custom date string into a datetime object
        last_time_in = parse_custom_date(last_date_str)
        # Calculate the current time
        current_time = datetime.now(last_time_in.tzinfo)
        # Calculate the time difference between the current time and the last entry time
        time_difference = current_time - last_time_in
        # Check if the last entry was made within the last minute
        if time_difference < timedelta(minutes=1):
            return False  # Entry not allowed within one minute interval
    return True  # Entry allowed
def parse_custom_date(date_str):
    # Example date string format: 'Tue Apr 16 2024 16:37:36 GMT+0500 (Pakistan Standard Time)'
    # Define the format string to parse the date with timezone information
    format_str = "%a %b %d %Y %H:%M:%S GMT%z (%Z)"
    try:
        # Parse the date string using the specified format
        parsed_date = datetime.strptime(date_str, format_str)
        return parsed_date
    except ValueError as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None
def record_attendance(person_name):
    # Get the database
    db = get_database()
    attendance = db["users"]
    # print(da)

    # Check if entry is allowed for this person within one minute interval
    if check_last_entry(person_name, attendance):
        # Generate a unique random ID for this attendance record
        attendance_id = str(uuid.uuid4())

        # Insert the attendance record with current time-in
        data = {
            "_id": attendance_id,
            "name": person_name,
            "email": "user@gmail.com",
            "attendance": 100,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        # attendance.insert_one(attendance_record)
        # print(f"Attendance recorded for {person_name} at {attendance_record['date']}")



        try:
            # Send POST request to the endpoint with the data
            response = requests.post(url, json=data)

            # Check the response status code
            if response.status_code == 201:
                # print("User added successfully:")
                print(response.json())  # Print the response data
            else:
                print(f"Failed to add user. Status code: {response.status_code}")
                print(response.json())  # Print any error message from the server

        except requests.exceptions.RequestException as e:
            print(f"Error occurred: {e}")
    else:
        print(f"Entry for {person_name} not allowed within one minute interval")

class FacePipeline:
    def __init__(self):
        self.embeddings = list()
        self.backend_models = ['antelopev2', 'buffalo_l', 'buffalo_sc', 'buffalo_s']
        self.model = FaceAnalysis(name=self.backend_models[2], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0)
        self.cap = cv2.VideoCapture(0)
        self.known_face_encodings = []
        self.known_face_names = []
        self.encodings = dict()
        self.anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_128.onnx')
        self.face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
        # Resize frame for a faster speed
        self.frame_resizing = 1
    def load_embeddings(self):
        import os
        curr_dir = os.getcwd()
        print(curr_dir)
        folder_path = curr_dir + "/faces-for-recognition/"
        print(folder_path)
        names = os.listdir(folder_path)
        print(f'{len(names)} persons found for facial recognition')
        for name in names:
            print(name)
            images = os.listdir(folder_path + "/" + name)
            for image in images:
                img = cv2.imread(folder_path + "/" + name + "/" + image)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self.model.get(rgb_img)
                img_embeddings = [face.embedding for face in faces]
                # Get encoding
                if (img_embeddings):
                    img_embeddings = np.asarray(img_embeddings)
                    # Store file name and file encoding
                    self.known_face_encodings.append(img_embeddings)
                    self.known_face_names.append(name)
                    self.encodings[name] = img_embeddings


    def increased_crop(self, img, bbox : tuple, bbox_inc : float = 1.5):
        # Crop face based on its bounding box
        real_h, real_w = img.shape[:2]
        
        x, y, w, h = bbox
        w, h = w - x, h - y
        l = max(w, h)
        
        xc, yc = x + w/2, y + h/2
        x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
        x1 = 0 if x < 0 else x 
        y1 = 0 if y < 0 else y
        x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
        y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
        
        img = img[y1:y2,x1:x2,:]
        img = cv2.copyMakeBorder(img, 
                                y1-y, int(l*bbox_inc-y2+y), 
                                x1-x, int(l*bbox_inc)-x2+x, 
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img

    def prediction(self, image):
        if len(image) > 0:
            faces = self.detection(image)
            # print("Here are faces -->", faces)
            # print("Bounding boxes ---->", bounding_boxes)
            # frame = self.model.draw_on(frame, faces)
            result = self.recognition(faces, image)
            # result_frame = self.check_anti_spoof(image, result)

            if result is not None:
                # frame = cv2.resize(image, (1920, 1080)) 

                # print(result)
                frame = self.plot_boxes(image, result)

                # # creating thread
                # t1 = threading.Thread(target=self.ImageSave, args=(result, frame), name='t1')
                # # starting threads
                # t1.start()

                
                # # putting the FPS count on the frame
                # cv2.putText(frame, fps, (1100, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
                return frame
            else:
                return image
    def detection(self,frame):
        return self.model.get(frame)


    # def check_anti_spoof(self, frame, data):
    #     if data:
    #         for name, cord in data.items():
    #             cord = (cord // self.frame_resizing).astype(np.int32)
    #             x_shape, y_shape = frame.shape[1], frame.shape[0]
    #             x1, y1, x2, y2 = cord[0], cord[1], cord[2], cord[3]
    #             bgr = (0, 255, 0)
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
    #             cv2.putText(frame, name , (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    #         return frame
    #     else:
    #         frame

    def recognition(self, faces, image):
        recognised_faces = {}
        # bounding_boxes = []
        # for face in faces:
        #     # print(face['bbox'])
            # bbx = face['bbox'].flatten().astype(int)
            # pred = anti_spoof([increased_crop(image, bbx, bbox_inc=1.5)])[0]
            # score = pred[0][0]
            # label = np.argmax(pred)   
        #     # print(bbx, score, label)
        #     bounding_boxes.append((bbx, score, label))
        # return bounding_boxes
        if len(faces) > 0:
            img_embeddings = []
            img_bounding_boxes = []
            counter = 0
            for face in faces:
                bbx = face['bbox'].flatten().astype(int)
                pred = anti_spoof([increased_crop(image, bbx, bbox_inc=1.5)])[0]
                # score = pred[0][0]
                label = np.argmax(pred) 
                if label == 0:
                    img_embeddings.append(face.embedding)
                    img_bounding_boxes.append(face.bbox.astype(np.int32))
                else:
                    recognised_faces["Spoofed"] = face.bbox.astype(np.int32)
                    # counter += 1
            img_embeddings = np.asarray(img_embeddings)
            for embedding in img_embeddings:
                result = self.compute_similarity(
                    embedding
                )
                recognised_faces[result] = img_bounding_boxes[counter]
                # print("Person is recognised with name --->", result, img_bounding_boxes[counter])
                counter += 1
            # print(recognised_faces)
            return recognised_faces

    def compute_similarity(self, img_embedding):
        embeddings_mat = np.asarray(self.known_face_encodings)
        img_embedding = np.asarray(img_embedding)
        for name, embedding in self.encodings.items():
            a = np.matmul(embedding.flatten(), img_embedding.flatten())
            b = np.sum(np.multiply(embedding, embedding))
            c = np.sum(np.multiply(img_embedding, img_embedding))
            distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
            if distance < 0.50:
                record_attendance(str(name))
                return name
        return "Unknown"
    #Method for drawing the bouding boxes for detection in image
    def plot_boxes(self, frame, data):
        if data:
            for name, cord in data.items():
                if name == "Spoofed":
                    cord = (cord // self.frame_resizing).astype(np.int32)
                    # x_shape, y_shape = frame.shape[1], frame.shape[0]
                    x1, y1, x2, y2 = cord[0], cord[1], cord[2], cord[3]
                    bgr = (0, 0, 255)  # Red color for Spoofed
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                else:
                    cord = (cord // self.frame_resizing).astype(np.int32)
                    # x_shape, y_shape = frame.shape[1], frame.shape[0]
                    x1, y1, x2, y2 = cord[0], cord[1], cord[2], cord[3]
                    bgr = (0, 255, 0)  # Green color for non-Spoofed
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)


            return frame
        else:
            return frame


    # def ImageSave(self,data, frame):
    #     if data:
    #         curr_dir = os.getcwd()
    #         curr_dir = curr_dir.split("FACE-RECOGNITION")[0]
    #         images_path = curr_dir + "/Entry_Camera/"
    #         print(images_path)
    #         if not os.path.exists(images_path):
    #             os.makedirs(images_path)
    #         records = []
    #         for name, cord in data.items():
    #             cord = (cord // self.frame_resizing).astype(np.int32)
    #             x_shape, y_shape = frame.shape[1], frame.shape[0]
    #             x1, y1, x2, y2 = cord[0], cord[1], cord[2], cord[3]
    #             face_image = frame[y1:y2, x1:x2]
    #             time = str(datetime.datetime.now())
    #             if len(face_image) > 0:
    #                 cv2.imwrite(images_path + name + "_" + time + ".png", face_image)
    #                 records.append([time, name])
    #             else:
    #                 continue
    #         if not os.path.exists(curr_dir + "/Entry_Camera_Records.csv"):
    #             data_df = pd.DataFrame(records, columns=['Time', 'Person'])
    #             data_df.to_csv(curr_dir + "/Entry_Camera_Records.csv")
    #         else:
    #             data_df = pd.DataFrame(records)
    #             with open(curr_dir + "/Entry_Camera_Records.csv","a") as f:
    #                 data_df.to_csv(f,header=False,index=True)

    def check_anti_spoof(self, frame, data):
        if data:
            for name, cord in data.items():
                cord = (cord // self.frame_resizing).astype(np.int32)
                x_shape, y_shape = frame.shape[1], frame.shape[0]
                x1, y1, x2, y2 = cord[0], cord[1], cord[2], cord[3]
                print("Here is frame ", frame, x1, y1, x2, y2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bbox = self.face_detector([frame])
                print(bbox)
                if bbox.shape[0] > 0:
                    bbox = bbox.flatten()[:4].astype(int)
                else:
                    return None
                pred = self.anti_spoof([self.increased_crop(frame, cord, bbox_inc=1.5)])
                print("Prediction from Anti-Spoof -->", pred)
                # score = pred[0][0]
                # label = np.argmax(pred)  
                # print("Here is the score -->", score, label)
                if pred == False:
                    # print(score)
                    # if score > 10:
                    res_text = "Live "
                    print(res_text, x1, x2, y1, y2)
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, name , (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                    # color = COLOR_REAL
                    # frame = cv2.resize(frame, (1000, 600)) 

                    # cv2.imshow('Entry camera',frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break


                    # # else: 
                    # #     res_text = "unknown"
                    # #     print(res_text)
                    # #     # color = COLOR_UNKNOWN
                else:

                    res_text = "Spoofed image"

                    print(res_text, x1, x2, y1, y2)
                    if x1 and y1:
                        pass
                        # bgr = (0,0,255)
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
                        # cv2.putText(frame, res_text , (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, bgr, 3)
                        # color = COLOR_REAL
                        # frame = cv2.resize(frame, (1000, 600)) 

                        # cv2.imshow('Entry camera',frame)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
            return frame
        else:
            None

if __name__ == "__main__":
    # parsing arguments
    def check_zero_to_one(value):
        fvalue = float(value)
        if fvalue <= 0 or fvalue >= 1:
            raise argparse.ArgumentTypeError("%s is an invalid value" % value)
        return fvalue
    
    p = argparse.ArgumentParser(
        description="Spoofing attack detection on videostream")
    p.add_argument("--input", "-i", type=str, default=None, 
                   help="Path to video for predictions")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Path to save processed video")
    p.add_argument("--model_path", "-m", type=str, 
                   default="saved_models\AntiSpoofing_bin_1.5_128.onnx", 
                   help="Path to ONNX model")
    p.add_argument("--threshold", "-t", type=check_zero_to_one, default=0.5, 
                   help="real face probability threshold above which the prediction is considered true")
    args = p.parse_args()
    
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof(args.model_path)
    Obj = FacePipeline()
    Obj.load_embeddings()
    # Create a video capture object
    if args.input:  # file
        vid_capture = cv2.VideoCapture(args.input)
    else:           # webcam
        vid_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width, frame_height)
    # print('Frame size  :', frame_size)

    if not vid_capture.isOpened():
        print("Error opening a video stream")
    # Reading fps and frame rate
    else:
        fps = vid_capture.get(5)    # Get information about frame rate
        # print('Frame rate  : ', fps, 'FPS')
        if fps == 0:
            fps = 24
        # frame_count = vid_capture.get(7)    # Get the number of frames
        # print('Frames count: ', frame_count) 
    
    # videowriter
    if args.output: 
        output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
    # print("Video is processed. Press 'Q' or 'Esc' to quit")
    
    # process frames
    rec_width = max(1, int(frame_width/240))
    txt_offset = int(frame_height/50)
    txt_width = max(1, int(frame_width/480))
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if ret:
            # Resize frame to 1080x800
            frame = cv2.resize(frame, (800, 600))
            # predict score of Live face
            # pred = make_prediction(frame, face_detector, anti_spoof)
            pred = Obj.prediction(frame)
            # print(pred)
            # if face is detected
            # if pred is not None:
            #     # print("Bounding boxes for detected faces --->", pred[0])
            #     for face in pred:
            #         # print(face[0])
            #         x1, y1, x2, y2 = face[0]
            #         score = face[1]
            #         label = face[2]
            #         # print(score, label)
            #         # (x1, y1, x2, y2), label, score = pred
            #         if label == 0:
            #             if score > args.threshold:
            #                 res_text = "REAL      {:.2f}".format(score)
            #                 color = COLOR_REAL
            #             else: 
            #                 res_text = "unknown"
            #                 color = COLOR_UNKNOWN
            #         else:
            #             res_text = "Spoof Image      {:.2f}".format(score)
            #             color = COLOR_FAKE
                        
            #         # draw bbox with label
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), color, rec_width)
            #         cv2.putText(frame, res_text, (x1, y1-txt_offset), 
            #                     cv2.FONT_HERSHEY_COMPLEX, (x2-x1)/250, color, txt_width)
            
            # if args.output:
            #     output.write(frame)
            
            # if video captured from webcam
            if not args.input:
                cv2.imshow('Facial Recognition System', pred)
                key = cv2.waitKey(20)
                if (key == ord('q')) or key == 27:
                    break
        else:
            print("Streaming is Off")
            break

    # Release the video capture and writer objects
    vid_capture.release()
    output.release()