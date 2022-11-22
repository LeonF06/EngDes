from tkinter import *
import serial
import time

serialcomm = serial.Serial('COM3', 9600)
serialcomm.timeout = 1

root = Tk()
root.title("Optical Inspection")

def run():
    serialcomm.write("run".encode('utf-8'))
    time.sleep(0.5)
    serialcomm.close

def defect():
    serialcomm.write("def".encode('utf-8'))
    time.sleep(0.5)
    serialcomm.close

button_run = Button(root, text="Run Program", padx = 50, pady = 10, command=run, bg="green")
button_run.pack()

button_defect = Button(root, text="Defect", padx = 45, pady = 10, command=defect, bg="red")
button_defect.pack()

root.mainloop()
