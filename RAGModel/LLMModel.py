from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from RAGModel.promptTemplate import create_rag_chain
from RAGModel.embeddingModel import load_embedding_vectordb
from langchain_core.messages import HumanMessage

db = load_embedding_vectordb() #Loading VectorDB

retriever = db.as_retriever(search_kwargs={'k': 15}) #VectorDB as the retriever for model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5) #LLM Model 

rag_chain = create_rag_chain(llm, retriever) #create rag chain instance


def get_completion(conversation):
    #initialize chat history
    chat_history = []
    #parse the incoming request
    question = conversation.message #new chat message from user
    print(question)
    history = conversation.conversationState #chat conversations
    print(type(history))
    for i in range(1, len(history), 2): #loop through the conversations (assuming that one by one from user and AI bot)
        user_message = history[i].message
        ai_response = history[i+1].message
        chat_history.extend([HumanMessage(user_message), ai_response]) #storing the conversation in chat history
    
    #Get the response from rag chain
    rag_chain.invoke({"input": question, "chat_history": chat_history})
    result = rag_chain.invoke({"input": question, "chat_history": chat_history})["answer"]
    return result
