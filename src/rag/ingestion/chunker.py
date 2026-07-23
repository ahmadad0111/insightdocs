from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:

    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=100
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_pages(self, pages):

        chunks = []

        chunk_id = 0

        for page in pages:

            split_texts = self.splitter.split_text(
                page["text"]
            )

            for text in split_texts:

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page_number": page["page_number"],
                        "text": text
                    }
                )

                chunk_id += 1

        return chunks