from Tkinter import *
from core import process_files
from docx_utils import get_text, read_watson_meta, clean_texts, get_sents
import logging, datetime, os
from model.utils import load_vocab, get_feat_vectors, process_word
from model.wimp import WImpModel
from model.config import config

LOG_FOLDER = "logs"

class LogHandler(logging.Handler):
    # Adapted from Moshe Kaplan: https://gist.github.com/moshekaplan/c425f861de7bbf28ef06

    def __init__(self, root, text):
        logging.Handler.__init__(self)
        self.text = text
        self.text.configure(state='disabled')
        self.root = root

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text.configure(state='normal')
            self.text.insert(END, msg + '\n')
            self.text.configure(state='disabled')
            self.text.yview(END)
        
        self.text.after(0, append)
        self.root.update_idletasks()
        self.root.update()


class Window:

    def __init__(self, root):

        #drawing UI
        self.reference_text = ""
        self.hypothesis_text = ""

        main_container = Frame(root)
        self.reference_label = Label(main_container, text="Reference File:")
        self.reference_label.grid(row=1, column=0)
        self.reference_entry = Entry(main_container)
        self.reference_entry.grid(row=1, column=1) 

        self.ref_bbutton= Button(main_container, text="Open", command=self.browse_reference_file)
        self.ref_bbutton.grid(row=1, column=2)

        self.hypothesis_label = Label(main_container, text="Hypothesis File:")
        self.hypothesis_label.grid(row=2, column=0)
        self.hypothesis_entry = Entry(main_container)
        self.hypothesis_entry.grid(row=2, column=1) 

        self.hyp_bbutton= Button(main_container, text="Open", command=self.browse_hypothesis_file)
        self.hyp_bbutton.grid(row=2, column=2)

        self.cbutton= Button(main_container, text="Process", command=self.process_files)
        self.cbutton.grid(row=3, column=1)
        main_container.grid(row=0, column=0, padx=(10, 10), pady=(20, 10))

        sub_container = Frame(root,  borderwidth=1, relief="sunken")
        log_label = Label(sub_container, text="Logs:", bg="#e5e5e5")
        log_label.pack(side="top", fill="x", expand=True)
        scrollbar = Scrollbar(sub_container)
        self.log_text = Text(sub_container, state = 'disabled', yscrollcommand = scrollbar.set,
            wrap="word", width = 60, height= 2)

        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)
        sub_container.grid(row=1, column=0, padx=(10, 10), pady=(20, 10))

        timestamp = datetime.datetime.now().isoformat()
        handler = LogHandler(root, self.log_text)
        logging.basicConfig(filename=os.path.join(LOG_FOLDER, timestamp + ".log"),
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s')

        # Add the handler to logger
        self.logger = logging.getLogger()      
        self.logger.addHandler(handler)

        #loading the models
        self.logger.info("Initializing importance labeling model:")
        self.model = WImpModel(config, load_vocab(config.words_vocab_path), 
            get_feat_vectors(config.feats_file), self.logger)
        self.model.setup()
        self.vocab = load_vocab(config.words_vocab_path)
        self.word_processor = process_word(self.vocab)

        

    def browse_reference_file(self):
        from tkFileDialog import askopenfilename

        Tk().withdraw() 
        self.ref_filename = askopenfilename()

        self.reference_entry.delete(0, END)
        self.reference_entry.insert(0, self.ref_filename)

        if self.ref_filename.endswith('.docx'):
            #update: other checks to esure correct reference file is read
            self.logger.info('Reading file: %s' % self.ref_filename.split('/')[-1])
            try:
                self.reference_text = '\n'.join(clean_texts(get_sents(self.ref_filename)))
                assert (self.reference_text)
            except:
                self.logger.error('File structure not recognized!')
                return

            self.logger.info('File read sucessfully.')

        else:
            self.logger.error('File upload failed, only docx files accepted!')        

    def browse_hypothesis_file(self):
        from tkFileDialog import askopenfilename

        Tk().withdraw() 
        self.hyp_filename = askopenfilename()

        self.hypothesis_entry.delete(0, END)
        self.hypothesis_entry.insert(0, self.hyp_filename)

        if self.hyp_filename.endswith('.docx'):
            #update: other checks to esure correct hypothesis file is read
            self.logger.info('Reading file; %s' % self.hyp_filename.split('/')[-1][:25])
            try:
                self.hypothesis_text = clean_texts(read_watson_meta(self.hyp_filename))[0]
                assert (self.hypothesis_text)
            except:
                self.logger.error('File structure not recognized!')
                return

            self.logger.info('File read sucessfully.')

        else:
            self.logger.error('File upload failed, only docx files accepted!')  

    def process_files(self, config=None):
        if self.reference_text != "" and self.hypothesis_text != "":
            reference_file = {'filename': self.ref_filename, 'text': self.reference_text}
            hypothesis_file = {'filename': self.hyp_filename, 'text': self.hypothesis_text}
            process_files(self.model, reference_file, hypothesis_file, self.word_processor, self.logger)
        else:
            self.logger.error('Not enough file(s) uploaded!')

root = Tk()
root.resizable(width=False, height=False)
root.wm_title("Word Importance Labeler")
window = Window(root)
root.mainloop()  