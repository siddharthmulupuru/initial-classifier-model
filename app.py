from fastai.vision.all import *
import gradio as gr

# model.pkl was exported with this labeling function; it must exist before load_learner
def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')  # matches learn.dls.vocab == [False, True]; True = cat

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image()
label = gr.Label()

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch(share=True)