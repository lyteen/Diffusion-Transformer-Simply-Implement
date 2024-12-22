import tkinter as tk

# function set the running status
def on_closing():
    global running
    running = False

# init the windows
root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", on_closing) # close the windows

# set the windows title
root.title("Simple Diffuser")

# set the windows size and open position
root.geometry("1200x600+500+400")
# set the graphical layout
canvas = tk.Canvas(root, bg="#4392F1", height=2200, width=2000, bd=0, highlightthickness=0, relief="ridge") # layout over the window, set the layout color is shada blue, and remove the border

# set the backgroud
canvas.place(x=0, y=0)
background_image = tk.PhotoImage(file="assets/background.png")
background = canvas.create_image(600.0, 300.0,image=background_image)

# set the header title, color is lightblack
header = canvas.create_text(560.0, 91.0, text="Diffuser", fill="#303030", font=("Roboto-Bold", int(30.0)))

# set a entry field get the user's input
input_label = tk.Label(root, text="Enter Value:", bg="#4392F1", fg="white", font=("Roboto-Medium", int(10)),  bd=0, highlightthickness=0, relief="ridge")
input_label.place(x=466.0, y=470.0, width=90, height=45)
input_entry = tk.Entry(root, bg="white", fg="black", font=("Roboto-Medium", int(30)))

# set the menu
menubar = tk.Menu(root)
about = tk.Menu(menubar, tearoff=0)

menubar.add_cascade(label='About', menu=about)

# create the button
start_img = tk.PhotoImage(file=f"assets/start.png")
start = tk.Button(image=start_img, borderwidth=0, highlightthickness=0, relief="flat")
back_img = tk.PhotoImage(file="assets/end.png")
back = tk.Button(image=back_img, borderwidth=0, highlightthickness=0, relief="flat")
info = canvas.create_text(594.0, 535.0, text='''Click "Start" Begin Inference''', fill="#ECE8EF", font=("Roboto-Medium", int(16.0)))

# when started
back["state"] = "disabled"
