import re

class SwitchBoardTokenizer():

	def __init__():
		pass

	def tokenize(self, conv_file):
		sents = []
		for line in conv_file.readlines():
			meta = line.split()[:3]
			#time_start, time_end = map(float, meta[1:])
			text = self.clean_text(' '.join(line.split()[3:]))
			if text != '':
				sents.append(text)
		return sents


	def clean_text(self, text):
		#-[ok]ay, etc.
	    text = re.sub(r'-\[([^\]\s]+)\]([^\[\s]+)', '\\1\\2', text.lower())

		#th[ey]- , sim[ilar]- , etc.
	    text = re.sub(r'([^\[\s]+)\[([^\]\s]+)\]-', '\\1\\2', text)

	    #remove [vocablized-noise] tags
	    text = re.sub(r'\[vocalized-(.*?)\]', '', text)

	    #[laughter-yeah], [laughter-i], etc.
	    text = re.sub(r'\[[^]]*?-(.*?)\]', '\\1', text)

	    #[silence], [noise], etc.
	    text = re.sub(r'\[.*?\]', '', text)

	    #remove trailing -
	    text = re.sub(r'(\w)-([ \n])', '\\1\\2', text)
	    
	    #remove symbols like { and }
	    text = re.sub(r'[{}]', '', text)

	    #fix multiple spaces
	    text = re.sub(r'(\s)+', '\\1', text)

	    return text.strip()


class TedliumTokenizer():

	def clean_text(self, text):
		text = text.strip().lower()

		if text == "ignore_time_segment_in_scoring":
			return ''
		#fix spacing (it 's)
		text = re.sub(r'(\w+) \'', '\\1\'', text.lower())
		#fix multiple spaces
		text = re.sub(r'(\s)+', '\\1', text)

		return text

	def tokenize(self, conv_file):
		sents = []
		for line in conv_file.readlines():
			meta = line.split()[:6]
			text = self.clean_text(' '.join(line.split()[6:]))
			if text != '':
				sents.append(text)

		return sents

