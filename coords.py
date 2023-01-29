from utils import get_strip,clear

from leds import cycle_individual_lights



def main():
	
	import picamera2 as picam
	
	camera = picam.Picamera2()
	
	camera.start_and_capture_file("tst.jpg")
	
	# ~ cycle_individual_lights()
	
	
	
if __name__ == "__main__":
	main()
