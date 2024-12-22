import re
import cv2
import webbrowser
import tkinter as tk
import src.interface as interface
from PIL import Image, ImageTk
from src.dit_inference import model_inference

# check input value
def check_value(input_value: str) -> int:
    if re.match(r"^[0-9]$", input_value):  # regex for integers (including negative)
        value = int(input_value)
        return value        
    else:
        return None

# send value to model
def send_value(image_paths: list[str], model_path: str, input_value: str, time_step: int) -> None:
    # send value to model
    if input_value != None:
        # generate image
        model_inference(image_paths, model_path, input_value, time_step)
    else:
        return

# show image
def display_image(canvas: object, image_path: str) -> None:
    # check the delete the last image
    if hasattr(canvas, 'current_image_id'):
        canvas.delete(canvas.current_image_id)
    try:
        # open image
        image = Image.open(image_path)
        # convert the photo image to tk-compatible format
        photo = ImageTk.PhotoImage(image)
        canvas.image = photo # keep reference
        # create the image on the canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        # configure canvas to match image size
        canvas.config(width=photo.width(), height=photo.height())
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# function loop play each frame for video 
def display_frame(canvas: object, cap: cv2.VideoCapture) -> None:
    # get STATUS
    global STATUS
    if STATUS == "origin":
        ret, frame = cap.read()
        if ret:
            try:
                # resize the frame to match the canvas size
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()
                frame = cv2.resize(frame, (canvas_width, canvas_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=img)
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                canvas.image = photo # keep reference to loop play the video
                canvas.after(30, display_frame, canvas, cap) # 30 ms play each frame
            except Exception as e:
                print(f"Error displaying frame: {e}")
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # loop the video
            display_frame(canvas, cap)

# function play the user video
def play_video(canvas: object, video_path: str) -> None:
    status_playing("origin")
    # check the delete the last image
    if hasattr(canvas, 'current_image_id'):
        canvas.delete(canvas.current_image_id)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    # configure the video canvas size 
    canvas.pack()
    canvas.config(width = 800, height = 440)
    # play the video
    display_frame(canvas, cap)

def status_playing(info: str) -> None:
    global STATUS
    STATUS = info
    if STATUS == "playing":
        interface.back["state"] = "disabled"
        interface.start["state"] = "disabled"
        interface.canvas.itemconfig(interface.info, text="Inferring...")
    elif STATUS == "finished":
        interface.back["state"] = "normal"
        interface.start["state"] = "normal"
        interface.canvas.itemconfig(interface.info, text='''Inference Successful Click "Start" See The Image''')
    elif STATUS == "origin":
        interface.back["state"] = "disabled"
        interface.start["state"] = "normal"
        interface.canvas.itemconfig(interface.info, text="Playing The User Tutorials...")
    elif STATUS == "error":
        interface.back["state"] = "normal"
        interface.start["state"] = "normal"
        interface.canvas.itemconfig(interface.info, text="Input Must Keep 0 - 9")

# toggle generate image
def toggle_image(image_paths: list[str]) -> None:
    global IMAGE_INDEX
    if IMAGE_INDEX == 0:
        interface.canvas.itemconfig(interface.info, text='''Click "Start" You Can See The Inference Result''')
    elif IMAGE_INDEX == 1:
        interface.canvas.itemconfig(interface.info, text='''Image For All Saved At Outputs Folder. You Can Create Another One''')
    display_image(canvas, image_paths[IMAGE_INDEX]) # show first image
    IMAGE_INDEX = (IMAGE_INDEX + 1) % 2 # only need to show two image

def check_input_value() -> bool:
    global INPUT_VALUE
    input_value = interface.input_entry.get()
    if input_value == INPUT_VALUE:
        return False
    elif input_value == "":
        return False
    else:
        INPUT_VALUE = interface.input_entry.get()
        return True
    
# start button command
def create_image(image_paths: list[str], model_path: str, time_step: int) -> None:
    global INPUT_VALUE
    if check_input_value():
        status_playing("playing")
        # check the value
        input_value = check_value(INPUT_VALUE)
        if input_value:
            image_path.clear()
            send_value(image_paths, model_path, input_value, time_step) # model inference
            status_playing("finished")
        else:
            status_playing("error")
    else:
        # check image_paths saved
        if len(image_paths) > 0:
            status_playing("finished")
            toggle_image(image_paths) # change the IMAGE_INDEX
        else:
            interface.canvas.itemconfig(interface.info, text="Image Isnt Generated")

if __name__=='__main__':

    STATUS = "origin"
    INPUT_VALUE = ""
    IMAGE_INDEX = 0

    canvas = tk.Canvas(interface.root)
    video_path = "Outputs/tutorials.mp4"
    model_path = "assets/fineTuned_weights_epoch_360.safetensors"
    time_step = 1000
    image_path = []
    interface.root.resizable(False, False)

    play_video(canvas, video_path)
    interface.about.add_command(label="Author", command=lambda : webbrowser.open("https://github.com/lyteen"))
    interface.start.config(command=lambda: create_image(image_path, model_path, time_step))
    interface.back.config(command=lambda: play_video(canvas, video_path))
    interface.running = True

    while interface.running:
        interface.root.update()
        interface.start.place(x=854, y=460, width=172, height=58)
        interface.back.place(x=154, y=460, width=172, height=58)
        interface.input_entry.place(x=554, y=470, width=180) # place Entry to canvas. remind: the position dont over the bound
        interface.root.config(menu=interface.menubar)