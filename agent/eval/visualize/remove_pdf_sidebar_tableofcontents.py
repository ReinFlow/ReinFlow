import fitz  # PyMuPDF
doc = fitz.open("input.pdf")
doc.set_toc([])
doc.save("output_no_toc.pdf")
doc.close()
