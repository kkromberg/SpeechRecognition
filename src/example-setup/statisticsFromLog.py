import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help ="Input XML file, which should be the log output of the sprint code.")
args = parser.parse_args()

elapsedTime = 0
realTime = 0

numberOfEdits = 0
referenceLength = 0

tree = ET.parse(args.input)
root = tree.getroot()
corpus = root.find('corpus')
for recording in corpus.findall('recording'):
	for segment in recording.findall('segment'):

		# Get the values for the real time factor computation
		elapsedTime += float(segment.find('timer').find('elapsed').text)
		realTime += float(segment.find('real-time').text)

		# Get the values for the WER computation
		statistic = segment.find('evaluation').find('statistic')
		numberOfEdits += float(statistic.find('cost').text)
		#print statistic.find('cost').text
		referenceLength += float(statistic.findall('count')[1].text)
		#print statistic.findall('count')[1].text



print "Word Error Rate", numberOfEdits / referenceLength * 100
print "Real Time Factor", elapsedTime / realTime 
