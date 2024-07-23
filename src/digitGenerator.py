"""
create 28X28 handwriting digit image
"""

import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageDraw

class DigitCollector:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=200, height=200, bg='black')
        self.canvas.pack()
        
        self.button_save = tk.Button(root, text="Save", command=self.save)
        self.button_save.pack()
        
        self.button_clear = tk.Button(root, text="Clear", command=self.clear)
        self.button_clear.pack()
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (200, 200), "black")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='white')

    def save(self):
        digit = self.get_digit()
        if digit is not None:
            self.image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
            self.image.save(f"digit_{digit}.png")
            print(f"Image saved as digit_{digit}.png")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), "black")
        self.draw = ImageDraw.Draw(self.image)

    def get_digit(self):
        digit = simpledialog.askstring("Input", "Enter the digit (0-9):")
        if digit and digit.isdigit() and 0 <= int(digit) <= 9:
            return digit
        else:
            print("Invalid digit")
            return None

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitCollector(root)
    root.mainloop()
