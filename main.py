import os

from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, MarkdownTextSplitter


def main():
    """
    Main function to load, process, and search text documents using LangChain and HuggingFace embeddings.
    """
    # Initialize the embedding function using the HuggingFace model
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           show_progress=True)
    db3 = None
    if input("quieres refrescar la base de datos? y/n ").lower() == "y":
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
        # Load the document and split it into chunks
        loader = TextLoader("TEXTOPIZZA.txt")
        documents = loader.load()

        # Split the document into chunks of 1000 characters with a 10-character overlap
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        docs = text_splitter.split_documents(documents)



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
    if db3 is None:
        db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


    # Prompt the user for a search query; exit if no query is provided
    query = input("Enter the text you want to search: ")
    if query != "":
        print("No search query provided. Exiting...")
        exit()

        # Perform a similarity search on the database using the provided query and print the top 5 results
        docs = db3.similarity_search(query, k=5)
        for i in range(5):
            print(docs[i].page_content)
            print("\n\n")

    retriever = db3.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "KnowledgeBaseRetriever",
        "Search for documents in the knowledge base."
    )
    tools = [tool]

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(temperature=0)
    question = input("Enter the question you want to ask: ")
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Respond only in Spanish."),
            ("human", "{input}"),
            # Placeholders fill up a **list** of messages
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"input": question})
    print(response.get("output"))


if __name__ == "__main__":
    main()
