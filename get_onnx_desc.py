import onnx
import sys

def main(argc, argv):
	if argc < 1:
		print("Not enough arguments.")
		return
	filename = argv[1]
	model = onnx.load(filename)
	print(model)
	
if __name__ == "__main__":
	main(len(sys.argv), sys.argv)
