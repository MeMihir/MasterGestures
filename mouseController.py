from numpy import BUFSIZE
from pynput import mouse
from pynput.mouse import Controller, Button
from sys import platform
import os
from gestureClassifier import classifyGesture
mouse = Controller()
MODE = 'Point'

def get_screen_resolution():
    ''' OS independent way of getting screen resolution. Adapted from pyautogui. '''
    if platform == 'linux':
        from Xlib.display import Display
        _display = Display(os.environ['DISPLAY'])
        return (_display.screen().width_in_pixels, _display.screen().height_in_pixels)
    
    elif platform == 'darwin':
        try:
            import Quartz
        except:
            assert False, "You must first install pyobjc-core and pyobjc"
        return (Quartz.CGDisplayPixelsWide(Quartz.CGMainDisplayID()),
                Quartz.CGDisplayPixelsHigh(Quartz.CGMainDisplayID()))
    
    elif platform == 'win32':
        import ctypes
        return (ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))

screenRes = get_screen_resolution()


def move_cursor(x,y):
    mouse.position = (x*1.2*screenRes[0],y*1.2*screenRes[1])

def rest():
    global MODE
    if MODE=='LClick':
        mouse.release(Button.left)
    elif MODE=='RClick':
        mouse.release(Button.left)
    elif MODE=='Scroll':
        mouse.release(Button.middle)
    MODE = 'Point'

def leftClick():
    global MODE
    if MODE!='LClick':
        mouse.press(Button.left)
    MODE = 'LClick'

def scroll():
    global MODE
    # if MODE!='Scroll':
    #     mouse.press(Button.middle)
    MODE = 'Scroll'

def mouse_control(points):
    global MODE
    if points != None and points.multi_hand_landmarks != None:
        gesture = classifyGesture(points)
        move_cursor(
            points.multi_hand_landmarks[0].landmark[8].x, 
            points.multi_hand_landmarks[0].landmark[8].y
        )
        print(gesture)
        if gesture=='Rest' or gesture == 'Point':
            rest()
        elif gesture=='RRightClick':
            leftClick()
        elif gesture=='Scroll':
            scroll()

    
def test():
    import cv2
    from handKeypoint import detectKeypoints, drawSkeleton
    from gestureClassifier import classifyGesture

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
            mouse_control(points)
            
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    finally:
        cap.release()

if __name__ == '__main__':
    test()