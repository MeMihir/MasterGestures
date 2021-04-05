import numpy as np
import xgboost as xgb
import pickle as pkl

GestureMapping = ['RDoubleClick', 'RLeftClick', 'RPoint', 'RRightClick', 'Scroll']
gestureClassifier = pkl.load(open('./models/gesture_classifier.pkl', 'rb'))

def preprocess_points(points):
    points = points.multi_hand_landmarks[0].landmark
    row = []
    for pt in points:
        row.append(pt.x)
        row.append(pt.y)
        row.append(pt.z)
    row = np.expand_dims(np.array(row),0)
    return xgb.DMatrix(row)

def classifyGesture(points):
    points = preprocess_points(points)
    pred = np.round(gestureClassifier.predict(data=points)).astype(np.int64)
    return GestureMapping[pred[0]]