from ultralytics import YOLO
import cv2

# Load all models
knife_model = YOLO("knife.pt")
appliance_model = YOLO("appliance.pt")
person_model = YOLO("yolov8n.pt")

# Camera (0 laptop cam / 1 external cam)
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    knife_results = knife_model(frame)
    appliance_results = appliance_model(frame)
    person_results = person_model(frame)

    knife_detected = False
    appliance_detected = False
    person_detected = False

    # Knife Detection
    for r in knife_results:
        if r.boxes is not None:
            for box in r.boxes:
                if float(box.conf[0]) > 0.5:
                    knife_detected = True

    # Appliance Detection
    for r in appliance_results:
        if r.boxes is not None:
            for box in r.boxes:
                if float(box.conf[0]) > 0.5:
                    appliance_detected = True

    # Person Detection
    for r in person_results:
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = person_model.names[cls]

                if name == "person" and float(box.conf[0]) > 0.75:
                    person_detected = True

    # ALERT LOGIC
    if person_detected and knife_detected:
        print("🚨 PERSON WITH KNIFE DETECTED!")

    elif person_detected and appliance_detected:
        print("⚠ PERSON WITH APPLIANCE DETECTED!")

    elif knife_detected:
        print("⚠ KNIFE DETECTED!")

    elif appliance_detected:
        print("Appliance detected")

    elif person_detected:
        print("Person detected")

    # Show combined boxes
    k_frame = knife_results[0].plot()
    a_frame = appliance_results[0].plot()
    p_frame = person_results[0].plot()

    temp = cv2.addWeighted(k_frame, 0.4, a_frame, 0.4, 0)
    final = cv2.addWeighted(temp, 0.5, p_frame, 0.5, 0)

    cv2.imshow("Smart Surveillance System", final)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
