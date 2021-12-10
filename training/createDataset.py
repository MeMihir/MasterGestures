import cv2
import numpy as np
import pandas as pd
from handKeypoint import detectKeypoints, drawSkeleton
from glob import glob

def extractRow(points):
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

    return np.array(row)

def DFHeaders():
    headers = []
    for i in range(21):
        headers.append(f'L{i}X')
        headers.append(f'L{i}Y')
        headers.append(f'L{i}Z')
    return headers

def createDataset():
    className = input('Enter Class Name : ')
    cap = cv2.VideoCapture(0)
    class_mat = np.zeros((1,63))

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            points = detectKeypoints(frame)
            drawSkeleton(frame, points)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            if points is  None or points.multi_hand_landmarks is None: 
                continue
            row = extractRow(points)
            class_mat= np.vstack((class_mat, np.expand_dims(row, 0)))
        print(class_mat.shape)
    except:
        print('Keyboard Interrupt')
    finally:
        cap.release()
        class_df = pd.DataFrame(class_mat[60:,:], columns=DFHeaders())
        class_df['class'] = [className]*len(class_df)
        class_df.to_csv(f'./training/data/{className}_df.csv')
        return class_df

def mergeMainDF(class_df):
    df = pd.read_csv('./training/data/mainDF.csv', index_col=0)
    df = pd.concat([df, class_df])
    df.to_csv('./training/data/mainDF.csv')


if __name__ == '__main__':
    class_df = createDataset()
    if input('Finalize Data [y/n]') == 'y':
        mergeMainDF(class_df)
    