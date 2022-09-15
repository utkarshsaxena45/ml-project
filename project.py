import streamlit as st
from cv2 import cv2
import numpy as np
st.set_page_config(page_title="Face Detection System",
page_icon="5985970.png")
choice=st.sidebar.selectbox("My Menu",("HOME","IMAGE"))
detectface=cv2.CascadeClassifier('face.xml')
st.title("Face Detection System")
if(choice=="HOME"):
	st.header("WELCOME")
	st.image("https://tenor.com/view/face-recognition-gif-19358861.gif")
elif(choice=="IMAGE"):
	img=st.file_uploader("Upload Image")
	if img:
		bytes=img.getvalue()
		img=cv2.imdecode(np.frombuffer(bytes,np.uint8),cv2.IMREAD_COLOR)
		face=detectface.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
		for (x,y,w,h) in face:
			img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
			st.image(img,channels='BGR')