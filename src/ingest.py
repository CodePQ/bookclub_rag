from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path):
    """
    Reads a PDF file and returns the full text as a single string.
    """
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            # Note: extract_text() is the official method. _extract_text is private.
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
        print(
            f"Successfully extracted {len(text)} characters from {file_path}.")
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits long text into smaller, overlapping chunks.
    RecursiveCharacterTextSplitter is great because it tries to split on
    logical breaks (\n\n, \n, " ") to keep meaning intact.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} text chunks.")
    return chunks


if __name__ == "__main__":
    # A simple test to see if our functions work!
    pdf_path = "data/the-adventure-of-the-cheap-flat.pdf"

    # 1. Load the PDF
    raw_text = load_pdf(pdf_path)

    if raw_text:
        # 2. Chunk the text
        chunks = chunk_text(raw_text)

        # 3. See the first chunk
        if chunks:
            print("\n--- FIRST CHUNK PREVIEW (First 500 chars) ---")
            print(chunks[0][:500])
            print("\n--------------------------")
