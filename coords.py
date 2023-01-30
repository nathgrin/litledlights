
import cv2
import time

from utils import get_strip,clear

from leds import blink_binary

def show_wait_and_capture(strip,cam,dt:float=0.001,grayscale: bool=True, fname: str="tmp.png"):
	
	strip.show()
	
	time.sleep(dt)
	
	ret,frame  = cam.read()
	if grayscale:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		
	cv2.imwrite( fname, frame )
		
	return ret,frame
	
def tst():

	cam = cv2.VideoCapture(0)
	
	cv2.namedWindow("test")
	
	img_counter = 0
	
	while True:
	    ret, frame = cam.read()
	    if not ret:
	        print("failed to grab frame")
	        break
	    cv2.imshow("test", frame)
	
	    k = cv2.waitKey(1)
	    if k%256 == 27:
	        # ESC pressed
	        print("Escape hit, closing...")
	        break
	    elif k%256 == 32:
	        # SPACE pressed
	        img_name = "opencv_frame_{}.png".format(img_counter)
	        cv2.imwrite(img_name, frame)
	        print("{} written!".format(img_name))
	        img_counter += 1
	
	cam.release()
	cv2.destroyAllWindows()

def tst2():
	loc = "_tmp/"
	vcap = cv2.VideoCapture(0)
	
	
	
	with get_strip() as strip:
		
		
		print(" - Frame off")
		ret, frame = vcap.read()
		cv2.imwrite(loc+"im_off.png", frame)
		
		
		print(" - Frame on")
		strip.fill( (255,255,255) )
		strip.show()
		for t in range(1050):
			ret, frame = vcap.read()
			if ret and t % 1000 == 0:
				cv2.imwrite(loc+"im_on.jpg".format(t), frame)
			time.sleep(0.001)
			
			
		vcap.release()
		
	

def main():
	# ~ tst2()
	# ~ exit()
	cam = cv2.VideoCapture(0)
	# Settings
	loc = "_tmp/"
	# ~ cam.set(38,1)
	# Set auto exposure to false
	# ~ cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
	# ~ exposure = 0.5
	# ~ cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
	
	cv2.namedWindow("tst")
	
	with get_strip() as strip:
		
		try:
			print(" - Off image")
			# ~ strip.fill( (255,255,255) )
			strip.fill( (0,0,0) )
			
			ret,im_off = show_wait_and_capture(strip,cam,fname=loc+"im_off.png")
			
			print(" - On  image")
			strip.fill( (255,255,255) )
			# ~ strip.fill( (0,0,0) )
			strip.show()
			# For some stupid reason we have to do this wait kind of thing
			# Well its stupid and it takes long
			print("   wait for it ...")
			for t in range(1000):
				ret,frame = cam.read()
				time.sleep(0.001)
			
			ret,im_on = show_wait_and_capture(strip,cam,fname=loc+"im_on.png")
			
			subtraction = cv2.subtract( im_on,im_off )
			cv2.imwrite( loc+"im_subtract.png", subtraction )
			
			cv2.imshow( "tst", subtraction )
			cv2.waitKey(0)
			
		finally: 
			cam.release()
			cv2.destroyAllWindows()
	
if __name__ == "__main__":
	main()
