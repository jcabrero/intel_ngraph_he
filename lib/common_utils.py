import resource
import multiprocessing as mp
import pickle
import csv
import time

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