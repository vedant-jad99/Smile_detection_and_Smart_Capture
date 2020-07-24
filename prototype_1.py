#!/usr/bin/env python3.8

import numpy as np 
import matplotlib.pyplot as plt 
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import os

semaphore = 1; frames = 0
cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("miscelleanous/haarcascade_frontalface_default.xml")
model = load_model(os.path.join(os.getcwd(), 'smile_detect'))
def show_web_cam():
    global semaphore, frames
    text = "miscelleanous/not_smiling.jpeg"
    ret, img = cap.read()
    if ret != None:
        img = cv2.resize(img, (630, 490))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_mod = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        copy_img = img_mod.copy()
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) != 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(img_mod, (x, y), (x + w, y + h), (0, 255, 255), 1)
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            roi = np.reshape(roi, (1, 64, 64, 1))/255.0
            pred = model.predict(roi)
            if pred[0][0] < pred[0][1]:
                cv2.putText(img_mod, "Smiling", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0))
                text = "miscelleanous/smiling.jpeg"
                frames += 1
                if semaphore == 1 and v.get() == 1 and frames > 20:
                    capture(copy_img)
            else:
                cv2.putText(img_mod, "Not smiling", (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
                text = "miscelleanous/not_smiling.jpeg"

        emoji = ImageTk.PhotoImage(Image.open(text))
        Pred_res = tk.Label(frame1, image=emoji, padx=10, pady=10, relief=tk.RAISED, borderwidth=5)
        Pred_res.image = emoji
        Pred_res.configure(image=emoji)
        Pred_res.grid(row=0, column=0)
        
        image = Image.fromarray(img_mod)
        image = ImageTk.PhotoImage(image)
        img_win = tk.Label(frame2, image=image, padx=10, pady=10)
        img_win.image = image
        img_win.configure(image=image)
        img_win.grid(row=0, column=0)
        img_win.after(30, show_web_cam) 

def capture(img):
    def close():
        global semaphore, frames
        capture_window.destroy()
        semaphore = 1
        frames = 0
    
    def save(img):
        filename = name.get()
        if ".jpg" in filename:
            cv2.imwrite(os.path.join(capture_path, filename), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            capture_window.destroy()
            done()
        else:
            error_msg = tk.Label(capture_window, text="ErrorSaving :   .jpg is missing from file name", width=40, padx=10, pady=10, fg='red')
            error_msg.grid(row=3,column=0)

    global semaphore
    semaphore = 0
    capture_window = tk.Toplevel()
    image = ImageTk.PhotoImage(Image.fromarray(img))
    image_win = tk.Label(capture_window, image=image, padx=10, pady=10, relief=tk.RAISED, borderwidth=5)
    image_win.image = image
    frame = tk.Frame(capture_window, width=60)
    text = tk.Label(capture_window, text="You look good in this. Want to capture it? :p", fg='red', padx=10, pady=20)
    capture_button = tk.Button(frame, text='Save', width=10, padx=10, relief=tk.GROOVE, borderwidth=5, command=lambda: save(img))
    close = tk.Button(frame, text="Cancel", width=10, padx=10, relief=tk.GROOVE, borderwidth=5, command=close)
    name = tk.Entry(frame, width=20, relief=tk.SUNKEN, borderwidth=5)
    title = tk.Label(frame, width =20, text="Name of file(with .jpg)", padx=10)

    image_win.grid(row=0, column=0)
    text.grid(row=1, column=0)
    frame.grid(row=2, column=0)
    capture_button.grid(row=0, column=0)
    name.grid(row=0, column=2)
    title.grid(row=0, column=1)
    close.grid(row=0, column=3)

def done():
    def complete():
        global semaphore, frames
        done_window.destroy()
        semaphore = 1; frames = 0
    done_window = tk.Toplevel()
    text = tk.Label(done_window, text="Picture saved successfully. Way to go!!! :D", width=40, padx=10, pady=5)
    ok = tk.Button(done_window, text="Done", width=5, padx=10, pady=5, command=complete)
    text.grid(row=0, column=0)
    ok.grid(row=1, column=0)

def exit():
    root.quit()

if __name__ == "__main__":
    capture_path = os.path.join(os.getcwd(), 'Captures')
    if os.path.exists(capture_path):
        pass
    else:
        os.mkdir(capture_path)
    root = tk.Tk()
    v = tk.IntVar()
    frame1 = tk.Frame(root, padx=10, pady=10)
    frame2 = tk.Frame(root, padx=10, pady=10, relief=tk.GROOVE, borderwidth=5)
    button = tk.Button(frame1, text="Close", width=20, command=exit)
    text_ = tk.Label(frame1, text="Smart Capture: ", width=20, pady=10)
    capture_off = tk.Radiobutton(frame1, text="Off", variable=v, value=0, width=10)
    capture_on = tk.Radiobutton(frame1, text="On", variable=v, value=1, width=10)

    frame1.grid(row=0, column=0)
    frame2.grid(row=0, column=1)
    button.grid(row=1, column=0)
    text_.grid(row=2, column=0)
    capture_off.grid(row=3, column=1)
    capture_on.grid(row=4, column=1)

    show_web_cam()
    root.mainloop()