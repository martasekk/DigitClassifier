import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

class NumberClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Number Classifier")
        self.geometry("540x640")
        self.resizable(False, False)

        #############################
        ######## Main window ########
        
        self.grid_rowconfigure(0, weight=1)  # top drawing area
        self.grid_rowconfigure(1, weight=0)  # bottom buttons + numbers
        self.grid_columnconfigure(0, weight=1)

        ######## Drawing Area #######
        
        self.draw_frame = ctk.CTkFrame(self, fg_color="#181818")
        self.draw_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.draw_frame.grid_rowconfigure(0, weight=1)
        self.draw_frame.grid_columnconfigure(0, weight=1)

        # Create blank image
        self.img_width, self.img_height = 500, 500
        self.img = Image.new("RGB", (self.img_width, self.img_height), "white")
        self.draw = ImageDraw.Draw(self.img)
        self.ctk_img = ctk.CTkImage(light_image=self.img, size=(self.img_width, self.img_height))

        # Area (label) to draw on
        self.label = ctk.CTkLabel(self.draw_frame, image=self.ctk_img, text="")
        self.label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.label.bind("<Button-1>", self.start_draw)
        self.label.bind("<B1-Motion>", self.paint)
        self.label.bind("<ButtonRelease-1>", self.predict_number)
        self.last_x, self.last_y = None, None


        ######## Lower area #########
        
        self.lower_frame = ctk.CTkFrame(self, fg_color="#2b2b2b")
        self.lower_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.lower_frame.grid_columnconfigure(0, weight=1, uniform="half")
        self.lower_frame.grid_columnconfigure(1, weight=1, uniform="half")

        # Clear drawing area
        self.clear_button = ctk.CTkButton(self.lower_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.clear_button.grid_columnconfigure(0, weight=1)

        # Area to display predicted number
        self.numbers_frame = ctk.CTkFrame(self.lower_frame)
        self.numbers_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.numbers_frame.grid_columnconfigure(0, weight=1)
        
        self.pred_label = ctk.CTkLabel(self.numbers_frame, text="Draw a number", font=ctk.CTkFont(size=24))
        self.pred_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Try to load model, if not found train a new one
        try:
          self.model = load_model("mnist_model.keras")
        except:
          self.model = self.train_model()
          
          
    ###########################
    ###### Drawing logic ######
    
    def start_draw(self, event):
        # Save starting point
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        # Drawing brush
        r = 15
        x, y = event.x, event.y
        
        # Draw ellipse
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill="black", outline="black")
        # Draw line (if fast movement)
        self.draw.line((self.last_x, self.last_y, x, y), fill="black", width=r*2+1)
        self.last_x, self.last_y = x, y
        self.ctk_img.configure(light_image=self.img)

    def clear_canvas(self):
        # Clear drawing area
        self.img.paste("white", [0, 0, self.img.size[0], self.img.size[1]])
        self.ctk_img.configure(light_image=self.img)
        self.pred_label.configure(text="Draw a number")

    ############################
    ###### PREDICT METHOD ######
    
    def predict_number(self, event):
        # Convert to grayscale
        img_gray = self.img.convert("L")
        # Invert colors
        img_inverted = Image.eval(img_gray, lambda x: 255 - x)
        # Crop bounding box
        bbox = img_inverted.getbbox()
        if bbox:
            cropped = img_inverted.crop(bbox)
        else:
            cropped = img_inverted

        # Resize to 20x20 for margin
        cropped.thumbnail((20,20), Image.LANCZOS)

        # Paste into centered 28x28 canvas
        new_img = Image.new("L", (28,28), 0)
        offset_x = (28 - cropped.width)//2
        offset_y = (28 - cropped.height)//2
        new_img.paste(cropped, (offset_x, offset_y))

        # Normalize and reshape
        img_array = np.array(new_img)/255.0
        img_array = img_array.reshape(1,28,28)

        # Predict
        prediction = np.argmax(self.model.predict(img_array))
        self.pred_label.configure(text=f"Predicted: {prediction}")

    ############################
    ###### Model training ######
    
    def train_model(self):
        # Load mnist data and train model
        numbers = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = numbers.load_data()
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(x_train, y_train, epochs=6, verbose=1)
        val_loss, val_acc = self.model.evaluate(x_test, y_test)
        print(f"Validation accuracy: {val_acc:.2f}, Validation loss: {val_loss:.2f}")
        self.model.save("mnist_model.keras")
        return self.model

###### Run app ######
if __name__ == "__main__":
    app = NumberClassifierApp()
    app.mainloop()
