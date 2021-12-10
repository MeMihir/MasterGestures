import numpy as np
import xgboost as xgb
import pickle as pkl

# GestureMapping = ['RDoubleClick', 'RLeftClick', 'RPoint', 'RRightClick', 'Scroll', 'Rest']
GestureMapping = ['RDClick', 'RLClick', 'RPoint', 'RRClick', 'RScroll']
gestureClassifier = pkl.load(open('./models/gesture_classifier3.pkl', 'rb'))

def preprocess_points(points):
    points = points.multi_hand_landmarks[0].landmark
    row = []
    minx = miny = 2
    maxx = maxy = 0.00001
    
    for pt in points:
        minx = min(minx, pt.x)
        miny = min(miny, pt.y)
        maxx = max(maxx, pt.x)
        maxy = max(maxy, pt.y)

    for pt in points:
        row.append((pt.x - minx)/(maxx-minx))
        row.append((pt.y - miny)/(maxy-miny))
        row.append(pt.z)

    row = np.expand_dims(np.array(row),0)
    return xgb.DMatrix(row)

def classifyGesture(points):
    points = preprocess_points(points)
    pred = np.round(gestureClassifier.predict(data=points)).astype(np.int64)
    return GestureMapping[pred[0]]

def test():
    import cv2
    from handKeypoint import detectKeypoints, drawSkeleton

    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            points = detectKeypoints(frame)
            drawSkeleton(frame, points)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            if points != None and points.multi_hand_landmarks != None:
                print(classifyGesture(points))
            
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    finally:
        cap.release()

if __name__ == '__main__':
    test()