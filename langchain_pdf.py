"""
This Telegram bot handles user-uploaded PDF files, extracts text from them, and uses LangChain and OpenAI's model
for question-answering. It employs two different tools (PDF Search and Google Search) to find answers based
on the contents of PDFs or perform web searches. The goal is to provide users with accurate and relevant answers
to their questions using a combination of local PDF content and external online search.

For instance, use these questions for the 'Little Prince' book:
Who is the narrator of The Little Prince?
[the pilot]
The six-year-old child gave up being an artist and chose to become what?
[A pilot]
What is the subject of the drawing in Chapter 1?
[a boa constrictor]
What is the name of the prince's home planet?
[Asteroid B-612.]
What should a person who flies planes study, according to the narrator?
[Geography]

"""

import os
from datetime import datetime

import telebot
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import GoogleSearchAPIWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from prompts import google_search_tool_prompt

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
PDF_PATH = "downloads"
LLM = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo"
)

bot = telebot.TeleBot(TOKEN)


def find_latest_pdf(user_id: int):
    """
    Finds the latest PDF file associated with a user based on their user ID.
    Args:
        user_id (int): The user's unique identifier.
    Returns:
        str or None: The path of the latest PDF file, or None if no matching file is found.
    """

    latest_document = None
    latest_date = None

    file_names = [file_name for file_name in os.listdir(PDF_PATH)]
    for file_name in file_names:
        filename_parts = file_name.split(".")[0].split("_")
        filename_user_id = filename_parts[0]

        if filename_user_id == str(user_id) and file_name.endswith(".pdf"):
            date = filename_parts[1] + "_" + filename_parts[2]
            datetime_object = datetime.strptime(date, "%Y-%m-%d_%H-%M-%S")

            if latest_date is None or datetime_object > latest_date:
                latest_document = file_name
                latest_date = datetime_object

    if latest_document:
        return f"{PDF_PATH}/{latest_document}"


def pdf_extract(user_id: int):
    """
    Extracts text from the latest PDF file associated with a user.
    Args:
        user_id (int): The user's unique identifier.
    Returns:
        list of str: List of text chunks extracted from the PDF.
    """

    pdf_file_path = find_latest_pdf(user_id)
    if pdf_file_path:
        with open(pdf_file_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks


def langchain_embeddings(question: str, user_id: int):
    """
    Generates an answer to a user's question using LangChain's question-answering capabilities.
    Args:
        question (str): The user's question.
        user_id (int): The user's unique identifier.
    Returns:
        str: The generated answer to the user's question.
    """

    # Create the vector DB
    chunks = pdf_extract(user_id)
    if chunks:
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # interact with OpenAI
        docs = knowledge_base.similarity_search(question)
        chain = load_qa_chain(LLM, chain_type="stuff")
        response = chain.run(input_documents=docs, question=question)
    else:
        response = "Please, upload your PDF document."

    return response


@bot.message_handler(commands=["start"])
def start(message: str):
    """
    Handles the '/start' command by sending a welcome message to the user.
    Args:
        message (telebot.types.Message): The incoming message object.
    """

    bot.send_message(message.chat.id, "Welcome to the LangChain Telegram bot! "
                                      "I can answer your questions about your PDF files. "
                                      "Just download a PDF file and type in a question "
                                      "and I'll do my best to answer it.")


@bot.message_handler(content_types=["document"])
def handle_message_document(message: str):
    """
    Handles incoming document messages (uploaded PDF files) by saving the file and notifying the user.
    Args:
        message (telebot.types.Message): The incoming message object.
    """

    # Generate a unique file name
    user_id = message.from_user.id
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{PDF_PATH}/{user_id}_{date}.pdf"

    # Get the PDF from the user
    file_info = bot.get_file(message.document.file_id)
    file_format = file_info.file_path.split(".")[-1]
    if file_format == "pdf":
        with open(file_name, 'wb') as f:
            file = bot.download_file(file_info.file_path)
            f.write(file)
            response = "Now you can ask questions to your PDF"
    else:
        response = f"Please, upload a PDF document instead of {file_format.upper()}"

    bot.send_message(message.chat.id, response)


@bot.message_handler(content_types=["text"])
def handle_message_text(message: str):
    """
    Handles incoming text messages by processing user questions and generating responses.
    Args:
        message (telebot.types.Message): The incoming message object.
    """

    user_id = message.from_user.id
    question = message.text

    # PDF tool
    pdf_search_tool = Tool(
        name="PDF Search",
        description="Useful for finding answers to questions in the PDF file. Use these answers first. "
                    "Only if there isn't any information for the question, then use the 'Google Search'",
        func=lambda x: langchain_embeddings(question, user_id),
    )

    # Google Search tool
    search = GoogleSearchAPIWrapper()
    google_search_tool = Tool(
        name="Google Search",
        description="Useful for finding answers to questions if there isn't any information for the question after using 'PDF Search'.",
        func=search.run,
    )

    tools = [pdf_search_tool, google_search_tool]
    # conversational agent memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=3,
        return_messages=True
    )
    # create our agent
    conversational_agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=LLM,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        memory=memory
    )
    conversational_agent.agent.llm_chain.prompt.messages[0].prompt.template = google_search_tool_prompt

    response = conversational_agent(question)
    bot.send_message(message.chat.id, response["output"])


bot.polling()
