import streamlit as st

#%%

#Face Detection logic

import cv2
import os
import numpy as np

class Main:
    
    _DEBUG = False
    def __init__(self, debug=False):
        
        Main._DEBUG = debug
        if Main._DEBUG :
            cv2.namedWindow("DEBUG::face detection", cv2.WINDOW_AUTOSIZE)
        
        self.main()

    
    def extract_faces(self,img,face_detector):
        '''
        Function detects the face and returns the cropped face
        
        args :
            img : image on which to find the face
            classifier : harr face classifier
        return :
            list of cropped faces
        '''

        height, width, _ = img.shape
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(img)

        if Main._DEBUG :
            if faces is not None :
            # drawing the faces on the frame
                img_copy = img.copy()
                for i in range(faces.shape[0]):
                    cv2.rectangle(img_copy, faces[i,:4].astype(np.int16), (0,0,255), 2, cv2.LINE_AA) # face
                    cv2.circle(img_copy, faces[i,4:6].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # left eye
                    cv2.circle(img_copy, faces[i,6:8].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # right eye
                    cv2.circle(img_copy, faces[i,8:10].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # nose
                    cv2.circle(img_copy, faces[i,10:12].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # mouth edge left
                    cv2.circle(img_copy, faces[i,12:14].astype(np.int16), 2, (0,0,255), 1, cv2.LINE_AA) # mouth edge right
                    cv2.putText(img_copy, f'{int(faces[i,14]*100)}',faces[i,:2].astype(np.int16)-4,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA) # confidence level
                    cv2.imshow('DEBUG::face detection', img_copy)

            # # if no face is detected delete the debug window
            # elif cv2.getWindowProperty('DEBUG::face detection', 0) == -1 : cv2.destroyWindow('DEBUG::face detection')

        if faces is None : return
        else :
            cropped_faces = []
            faces = faces.astype(np.int16)
            for i in range(faces.shape[0]):
                x,y,w,h = faces[i,:4]
                print(img[y:y+h,x:x+w].shape)
                cropped_faces.append(img[y:y+h,x:x+w])
            return cropped_faces

    # def align_face(self,face):
    #     pass

    def main(self):
        # Yunet face classifier
        directory = os.path.dirname(__file__)
        weights = os.path.join(directory, "face_detection_yunet_2022mar.onnx")
        face_detector = cv2.FaceDetectorYN_create(weights,"",(0,0))

        # # Initialize Webcam
        # video_capture = cv2.VideoCapture(0)

        # timer = cv2.TickMeter()
        # while True :
        #     timer.start()
            
            # Capture frame-by-frame
            # _, frame = video_capture.read()
        if 'pic_list' not in st.session_state:
            st.session_state.pic_list = []
            st.session_state.count=0

        frame = st.camera_input("Take a picture")
        if frame:
            # bytes_data = frame.getvalue()
            # IMG = cv2.imread(bytes_data)

            
            # To read image file buffer with OpenCV:
            bytes_data = frame.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)


            # img=st.image(frame)
            # np_img = cv2.imread('img.jpg')
            #np_img = np.array(jpg)
            #print(np_img)
            faces = self.extract_faces(cv2_img, face_detector)

            if faces is None:
                st.write("Previous Photo was not recorded Please Align Your Face Properly")
            else:
                st.session_state.pic_list.append(frame)
                st.session_state.count += 1

            st.write('Count = ', st.session_state.count)

            for i in  range(len(st.session_state.pic_list)):
                st.image(st.session_state.pic_list[i])





        #     timer.stop()

        #     # print Processing time on the screen
        #     cv2.putText(frame, f"FPS: {timer.getFPS():.2f}",(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        #     # Display the processed frame
        #     cv2.imshow('Face Detection', frame)

        #     key_pressed = cv2.waitKey(10)
        #     if key_pressed == ord('q') : break # press 'q' to exit
        #     if key_pressed == ord(' ') : pass # press 'space bar'

        # # release the capture
        # video_capture.release()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    Main()


#%%

#Streamlit to take image inputs




# if 'pic_list' not in st.session_state:
#    st.session_state.pic_list = []
#    st.session_state.count=0

# picture = st.camera_input("Take a picture")

# if picture:
#     st.session_state.pic_list.append(picture)
#     st.session_state.count += 1

# st.write('Count = ', st.session_state.count)

# for i in  range(len(st.session_state.pic_list)):
#     st.image(st.session_state.pic_list[i])



# st.title('Counter Example')
# if 'count' not in st.session_state:
#     st.session_state.count = 0

# increment = st.button('Increment')
# if increment:  
#     st.session_state.count += 1

# st.write('Count = ', st.session_state.count)