from PyPDF2 import PdfReader #import pdf reader

reader = PdfReader("Swarna new.pdf") #open pdf

text = "" #create empty string

for page in reader.pages:
    text += page.extract_text()

print(text)
