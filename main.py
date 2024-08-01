import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, MarkdownTextSplitter

def main():
    """
    Main function to load, process, and search text documents using LangChain and HuggingFace embeddings.
    """

    # Load the document and split it into chunks
    loader = TextLoader("TEXTOPIZZA.txt")
    documents = loader.load()

    # Split the document into chunks of 1000 characters with a 10-character overlap
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    # Initialize the embedding function using the HuggingFace model
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                               show_progress=True)

    # Remove the persist_directory, if it exists, along with all its files
    if os.path.exists("./chroma_db"):
        os.system("rm -r ./chroma_db")

    # Create a Chroma database from the split documents and embeddings, and persist it
    Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

    input("vamos a cargar el segundo documento de markdown")

    # Load the second document (Markdown format)
    loadermd = TextLoader("MarkdownAUTO.md")
    documentsmd = loadermd.load()

    # Split the Markdown document into chunks of 1000 characters with a 10-character overlap
    text_splittermd = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=10)
    docsmd = text_splittermd.split_documents(documentsmd)

    # Reopen the Chroma database and add the new Markdown document chunks
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    db3.add_documents(docsmd)

    # Prompt the user for a search query; exit if no query is provided
    query = input("Enter the text you want to search: ")
    if query == "":
        print("No search query provided. Exiting...")
        exit()

    # Perform a similarity search on the database using the provided query and print the top 5 results
    docs = db3.similarity_search(query, k=5)
    for i in range(5):
        print(docs[i].page_content)
        print("\n\n")



if __name__ == "__main__":
    main()