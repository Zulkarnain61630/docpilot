from pypdf import PdfReader

reader = PdfReader("data/notes.pdf")
print("Pages:", len(reader.pages))

text = reader.pages[0].extract_text()
print("----- Page 1 text -----")
print(text[:800])
print("-----------------------")
print("Blank lines (paragraph breaks) found:", text.count("\n\n"))

from rag import chunk_text

chunks = chunk_text(text)
print("\nChunks on page 1:", len(chunks))
print("End of chunk 1:   ...", chunks[0][-80:])
if len(chunks) > 1:
    print("Start of chunk 2:", chunks[1][:80], "...")