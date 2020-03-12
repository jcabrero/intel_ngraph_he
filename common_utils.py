import resource
import multiprocessing as mp
import pickle
import csv
import time
import requests
import smtplib

def get_ram():
	kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	mb, kb = kb / 1024, kb % 1024
	gb, mb = mb / 1024, mb % 1024
	return "[%0.4f GB]" % (gb)

def get_elapsed_time(tic):
	elapsed = time.time() - tic
	mins, secs = int(elapsed / 60), elapsed % 60
	hours, mins = int(mins / 60), mins % 60
	return "[%d HOURS, %02d MINS, %02d SECS]" % (hours, mins, secs)

def pickle_object(obj, filename):
	with open(filename, 'wb') as f:
		pickle.dump(obj, f)

def unpickle_object(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
		return obj
##############################################################################################
# FUNCTION: gen_csv_from_tuples
# DESCRIPTION:  Generates a csv with all the links and the user who posted it.
# OUTPUT_FORMAT: (index, "AuthorId", "AuthorName", "Link")
##############################################################################################
def gen_csv_from_tuples(name, titles, rows):
	#file = open('id_user_url.csv', mode='w+')
	file = open(name, mode='w+')
	writer = csv.writer(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
	writer.writerow(titles)
	for row in rows:
		writer.writerow(row)
##############################################################################################
# FUNCTION: read_csv_as_list
# DESCRIPTION:  Generates a list of tuples of the CSV
# OUTPUT_FORMAT: (index, "AuthorId", "AuthorName", "Link")
##############################################################################################
def read_csv_list(name):
	with open(name) as f:
		data=[tuple(line) for line in csv.reader(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)]
		return data 

def join_all_results(self, origin_filename):
	tic = time.time()
	directory, filename = self.get_filename_dir(origin_filename)
	list_files = [directory + name for name in os.listdir(directory) if filename in name and not filename == name]
	result_file = origin_filename
	f1 = open(result_file, 'w+', buffering=2)
	first = True
	total = len(list_files)
	for ind, file in enumerate(list_files):
		print("Joining files", "[%d Files Processed]" %(ind), "[%0.3f Percentage]" % ((ind / total) * 100), get_ram(), get_elapsed_time(tic), end='\r')
		with open(file, 'r') as f2:
			line = f2.readline()
			if first:
				f1.write(line)
				f1.flush()
				first = False
			line = f2.readline()
			while line:
				f1.write(line)
				f1.flush()
				line = f2.readline()
		os.remove(file)
	f1.close()
	print("[END] Joining files", "[%d Files Processed]" %(ind), "[%0.3f Percentage]" % ((ind / total) * 100), get_ram(), get_elapsed_time(tic), end='\r')



def send_mail(message):
	# creates SMTP session 
	smtpserver = smtplib.SMTP('smtp.gmail.com', 587) 

	smtpserver.ehlo()

	# start TLS for security 
	smtpserver.starttls() 

	smtpserver.ehlo()
	  
	# Authentication 
	smtpserver.login(mail_username, mail_password) 
	  
	# message to be sent 
	#message = "Message_you_need_to_send"
	  
	# sending the mail 
	smtpserver.sendmail(mail_username, mail_dest, message) 
	  
	# terminating the session 
	smtpserver.quit() 