# test translation
from Translator import Translator

en_lines = open('en_test.txt').read().split('\n')
epoch_num=10
for epoch in range(epoch_num):
   model = Translator()
   modelfile = "en2ja-" + str(epoch) + ".model"
   model.load_model(modelfile)
   for i in range(len(en_lines)):
      en_words=en_lines[i].lower().split()
      ja_words=model.test(en_words)
      print("model",epoch,i+1,''.join(ja_words))