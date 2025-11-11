import faiss, numpy as np, time

def build_faiss_index(db_descriptors): index=faiss.IndexFlatL2(db_descriptors.shape[1]); index.add(db_descriptors.astype('float32')); return index

def search_index(index,qdesc,k=5): t0=time.time(); D,I=index.search(qdesc.astype('float32'),k); return D,I,(time.time()-t0)/max(1,qdesc.shape[0])
