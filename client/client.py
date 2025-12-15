import cv2
import requests
import base64
import numpy as np

SERVER_URL = "http://localhost:8000/pose/"

cap = cv2.VideoCapture(1)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break


    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
    

    res = requests.post(SERVER_URL, files=files)

    if res.status_code == 200:
        data = res.json()
        print(f"People detected: {data.get('persons')}")
        

        if data.get("persons", 0) > 0:
            print("Body Keypoints (first person):", data["body_keypoints"][0][:5])  
            print("Face Keypoints:", data["face_keypoints"])  
            print("Left Hand Keypoints:", data["left_hand"])  
            print("Right Hand Keypoints:", data["right_hand"])  
        

        rendered_image = data.get("rendered_image")
        if rendered_image:

            img_data = base64.b64decode(rendered_image)
            nparr = np.frombuffer(img_data, np.uint8)
            decoded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("Rendered Pose", decoded_image) 
        
    else:
        print("Server error:", res.text)


    cv2.imshow("Client Webcam", frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

