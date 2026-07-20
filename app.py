from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')

categories = ('Cat', 'Dog')  # double-check order matches your training notebook

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image()
label = gr.Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch()