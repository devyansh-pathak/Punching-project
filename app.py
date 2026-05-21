from flask import request , Flask ,jsonify , send_from_directory
from flask_cors import CORS
import os
from langchain_core.documents import Document
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.document_loaders.pdf import PyPDFLoader

os.environ['SENTENCE_TRANSFORMERS_HOME'] = './model_cache'
api_key = os.environ.get('GEMINI_API_KEY')


app=Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')


os.makedirs('uploads', exist_ok=True)


def loader(pdffile):
    pdf_loader = PyPDFLoader(pdffile)
    document = pdf_loader.load()
    all_docs=[]
    all_docs.extend(document)
    return all_docs
from langchain_text_splitters import RecursiveCharacterTextSplitter
def split_docs(document,chunk_size=500,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = None
    chunked_docs = text_splitter.split_documents(document)
    return chunked_docs



from sentence_transformers import SentenceTransformer
class EmbeddingManager:
    def __init__(self,model_name= "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        print("loading model....",self.model_name)
        self.model = SentenceTransformer(self.model_name)
        print("embedding dimensions = ",self.model.get_sentence_embedding_dimension())

    def generate_embeddings(self,text):
        embeddings = self.model.encode(text,show_progress_bar = True)
        print("embedding shape:",embeddings.shape)
        return embeddings
    def clear(self):
        embeddings=None





import uuid
import os
import chromadb
class vectorstore:
    def __init__(self,persist_directory = "data/vector_store", collection_name = "pdf_documents"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        self.client = None

        self. initialize_store()
    def initialize_store(self):
        os.makedirs(self.persist_directory,exist_ok=True)
        self.client = chromadb.PersistentClient(path = self.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            metadata = {"description":"vector store collection for pdf embeddings in RAG"}
        )

        print("inittialized the vector store with collection:", self.collection_name)
        print("docs in collection:", self.collection.count())

    def add_documents(self, documents , embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("num of docs doesnt match num of embeddings")
        ids=[]
        all_metadata=[]
        document_content=[]
        embedding_list=[]

        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            doc_id = f"doc_{uuid.uuid4()}"
            ids.append(doc_id)

            
            metadata = dict(doc.metadata)
            metadata["doc_index"]=i
            metadata["content_length"]=len(doc.page_content)
            all_metadata.append(metadata)

            document_content.append(doc.page_content)

            embedding_list.append(embedding.tolist())

            self.collection.add(
                ids=ids,
                metadatas = all_metadata,
                documents = document_content,
                embeddings = embedding_list
            )
        
        print("total document added in vector store=",len(document_content))
        print("docs in collection:", self.collection.count())
    def clear(self):
        self.client.delete_collection(self.collection_name)




class RAGretriever:
    def __init__(self,embedding_manager , vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
    def retrieve(self,query,top_k=5,score_threshold=0):
        query_embeddings = self.embedding_manager.generate_embeddings([query])[0]
        results=None
        results = self.vector_store.collection.query(
            query_embeddings = [query_embeddings.tolist()],
            n_results = top_k 
        )
        retrieved_docs=[]
        if results["documents"] and results["documents"][0]:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for i ,(doc_id,metadata,document,distance) in enumerate(zip(ids,metadatas,documents,distances)):
                similarity_score = 2- distance

                if similarity_score>=score_threshold:
                    retrieved_docs.append({
                        "ids":doc_id,
                        "document":document,
                        "metadata":metadata,
                        "distance":distance,
                        "similarity_score":similarity_score,
                        "rank":i+1
                    })
            print(f"retrieved {len(retrieved_docs)} documents")

        else:
            print("no document found")
        return retrieved_docs
    


from langchain_google_genai import ChatGoogleGenerativeAI
def llm_initialize(api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )
    return llm

def generate_output(query , retriever , llm , top_k=2):
    results = retriever.retrieve(query, top_k)
    context = "\n".join(doc["document"] for doc in results) if results else ""
    if not context:
        print("we found no context for given query")
    prompt = f"""You are an insurance document parser.
                Answer ONLY from the given context.
                Be extremely precise and concise.
                For policy type question: reply ONLY 'TP' or 'OD' or 'COMPREHENSIVE' nothing else.
                For amounts: reply ONLY the number, no text.
                  query:{query},
                  context:{context}"""
    response = llm.invoke([prompt.format(context = context ,  query = query)])
    return response.content
from concurrent.futures import ThreadPoolExecutor
def list_ans(rag_retriever,llm):
    queries = [
        "what is the company name only",
        "what is the total amount in rupees",
        "what is total premium before tax evaluation",
        "what is the Net OD Premium amount, return 0 if not applicable",
        "what is the TP (Third Party) premium amount only",
        "what is the vehicle type like Two Wheeler or Four Wheeler",
        "what is the total tax on policy",
        "what is the policy type, answer ONLY one of these: ONLY TP(Third party/ liability only), ONLY OD(Own damage), COMPREHENSIVE (TP and OD)",
        "What is the vehicle registration number, it follows format like DL1234AB5678 or MH-02AB-1234, return ONLY the registration number nothing else",
        "what is the name of Insured name",
        "what is the fuel of the vehicle chose from:( PETROL , CNG , DISEL , ELECTRIC ) IF vehicle type is two wheeler it is more likely to be PETROL",
        "what is the policy issue date write all dates in a similar format",
        "what is the name of broker choose from them mostly: POLICYBAZAAR , INSURANCE DEKHO , SHALINI PATHAK , KOMAL "
    ]
    def run(q):
        return generate_output(q, rag_retriever, llm)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, queries))

        return results
gemini_api_key = os.getenv("GEMINI_API_KEY")
embedding_manager=EmbeddingManager()
def punching(filename, associate, cr, cg,payment):
    all_docs=loader(f'uploads/{filename}')
    global embedding_manager
    chunk = split_docs(all_docs)
    vector_store = vectorstore()
    vector_store.clear()
    vector_store = vectorstore()
    texts = [doc.page_content for doc in chunk]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunk,embeddings)
    rag_retriever=RAGretriever(embedding_manager,vector_store)
    api_key = gemini_api_key
    llm=llm_initialize(api_key)
    ans_list=list_ans(rag_retriever,llm)
    df = pd.DataFrame([{
        'Policy issue Date':  ans_list[11], 
        'client name':        ans_list[9],
        'Registration no.':   ans_list[8],
        'Vehicle info':       ans_list[5],
        'Policy Type':        ans_list[7],
        'Fuel':               ans_list[10],
        'GVW/Seating capacity': "To be filled",             
        'Company':            ans_list[0],
        'online/offline':     "To be filled",
        'broker':             ans_list[12],
        'OD Premium':         ans_list[3],
        'TP Amount':          ans_list[4],
        'Total Tax':          ans_list[6],
        'Total Amount':       ans_list[1],
        'Commission premium':"To be filled",
        'Commision Recieved':  f"{cr} %",
        'Commission given':    f"{cg} %",
        'Associate':          associate,
        'payment_status':      payment,
        'Remarks':            "To be filled/No remarks "
    }])

    return jsonify({
        'status':  'ok',
        'columns': df.columns.tolist(),
        'rows':    df.values.tolist()       
    })
@app.route('/submit', methods=['POST'])
def submit():
    pdf       = request.files['pdf']
    associate = request.form['associate']
    cr        = request.form['cr']
    cg        = request.form['cg']
    payment = request.form['payment_status']

    pdf.save(f'uploads/{pdf.filename}')
    pdf_path = f'uploads/{pdf.filename}'

    result = punching(pdf.filename, associate, cr, cg, payment)
    os.remove(pdf_path) 
    return result 

import gspread
from google.oauth2.service_account import Credentials
import json
def save_to_sheet(record):
    creds_json = os.environ.get('GOOGLE_CREDENTIALS')
    creds_dict = json.loads(creds_json)
    creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
    
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
    )
    client = gspread.authorize(creds)
    sheet  = client.open("Punching").sheet1

    if sheet.row_count == 1 and sheet.row_values(1) == []:
        sheet.append_row(list(record.keys()))

    sheet.append_row(list(record.values()))


@app.route('/punch', methods=['POST'])
def punch():
    try:
        data      = request.get_json()
        record    = data['record']
        save_to_sheet(record)


        return jsonify({'status': 'ok'})

    except Exception as e:
        print("PUNCH ERROR:", str(e))      
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)