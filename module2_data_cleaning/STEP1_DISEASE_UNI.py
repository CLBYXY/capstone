import pandas as pd
import re,faiss,torch,pandas as pd,csv
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances


def disease_pipeline(input_file):

    # =========================
    # Step1: lowercase
    # =========================
    df = pd.read_excel(input_file)

    df["Type_of_illness"] = df["Type_of_illness"].str.lower().where(df["Type_of_illness"].notna())

    def extract_first_bracket(t):
        if pd.isna(t): return None
        m=re.search(r'\[(.*?)\]',str(t))
        if not m: return None
        return m.group(1).split(",")[0].strip().replace('"',"").replace("'","")

    df["disease_extracted"] = df["Type_of_illness"].apply(extract_first_bracket)


    # =========================
    # Step2: SapBERT + UMLS mapping
    # =========================
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model=AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)
    model.eval()

    def encode(texts,batch_size=32):
        vec=[]
        for i in tqdm(range(0,len(texts),batch_size)):
            b=[str(x) for x in texts[i:i+batch_size]]
            inp=tokenizer(b,padding=True,truncation=True,max_length=32,return_tensors="pt").to(device)
            with torch.no_grad(): e=model(**inp).last_hidden_state.mean(dim=1)
            vec.append(e.cpu())
        return torch.cat(vec,0).numpy().astype("float32")

    def fix_umls_term(t):
        if isinstance(t,str) and ";" in t:
            p=[x.strip() for x in t.split(";")]
            if len(p)==2: return p[1]+" "+p[0]
        return t

    def load_disease_cui(path):
        df=pd.read_csv(path,sep="|",header=None,dtype=str).iloc[:,:4]
        df.columns=["CUI","TUI","STN","STY"]
        return set(df[df["TUI"].isin(["T047","T191","T048"])]["CUI"])

    def load_umls_terms(path,cuis):
        df=pd.read_csv(path,sep="|",header=None,dtype=str,engine="python",quoting=csv.QUOTE_NONE,on_bad_lines="skip").iloc[:,:18]
        df.columns=["CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"]
        df=df[df["CUI"].isin(cuis)]
        df=df[(df["LAT"]=="ENG")&(df["TTY"]=="PT")]
        df=df[df["STR"].astype(str).str.len()>1]
        df["STR"]=df["STR"].apply(fix_umls_term)
        return df["STR"].tolist(),df["CUI"].tolist()

    mrsty_file="MRSTY.RRF"
    mrconso_file="MRCONSO.RRF"

    diseases=[extract_first_bracket(x) for x in df["Type_of_illness"]]
    df["disease_extracted"]=diseases

    disease_cuis=load_disease_cui(mrsty_file)
    umls_terms,cuis=load_umls_terms(mrconso_file,disease_cuis)

    umls_vec=encode(umls_terms);faiss.normalize_L2(umls_vec)
    index=faiss.IndexFlatIP(umls_vec.shape[1]);index.add(umls_vec)

    disease_vec=encode(diseases);faiss.normalize_L2(disease_vec)
    D,I=index.search(disease_vec,1)

    df["UMLS_term"]=[umls_terms[i[0]] for i in I]
    df["CUI"]=[cuis[i[0]] for i in I]
    df["similarity"]=[d[0] for d in D]


    # =========================
    # Step3: KMeans clustering
    # =========================
    terms=df["UMLS_term"].astype(str).tolist()

    embeddings=encode(terms)
    kmeans=KMeans(n_clusters=25,random_state=42,n_init="auto")
    labels=kmeans.fit_predict(embeddings)
    centers=kmeans.cluster_centers_

    dist=cosine_distances(centers,embeddings)
    center_terms=[terms[d.argmin()] for d in dist]

    df["cluster"]=labels
    cluster_names={i:center_terms[i] for i in range(25)}
    df["cluster_name"]=df["cluster"].map(cluster_names)


    # =========================
    # Final Output
    # =========================
    df.to_csv("STEP1-DISEASE_UNI.csv", index=False)