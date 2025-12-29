# RAG (Retrieval-Augmented Generation) 아키텍처 가이드

## 1. RAG란?

RAG는 **검색 증강 생성**의 약자로, LLM(Large Language Model)의 한계를 극복하기 위해 외부 지식을 검색하여 응답 생성에 활용하는 기술입니다.

### LLM의 한계점
- **지식 컷오프**: 학습 데이터 이후의 정보를 알지 못함
- **할루시네이션**: 사실이 아닌 정보를 생성할 수 있음
- **도메인 특화 지식 부족**: 특정 기업/분야의 내부 정보를 알지 못함

### RAG의 해결 방식
외부 데이터베이스에서 관련 정보를 검색 → LLM에 컨텍스트로 제공 → 정확한 답변 생성

---

## 2. RAG 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG 시스템 아키텍처                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────── INDEXING PIPELINE ─────────────────────┐   │
│  │                                                                  │   │
│  │   [Documents]  →  [Loader]  →  [Splitter]  →  [Embedding]       │   │
│  │       │              │            │              │               │   │
│  │    PDF, TXT,      문서 로드      청킹          벡터 변환          │   │
│  │    HTML, etc.                 (Chunking)                        │   │
│  │                                                    ↓             │   │
│  │                                            [Vector Store]        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                       │                 │
│                                                       ↓                 │
│  ┌───────────────────── RETRIEVAL & GENERATION ────────────────────┐   │
│  │                                                                  │   │
│  │   [User Query] → [Query Embedding] → [Similarity Search]        │   │
│  │        │                                    │                    │   │
│  │        │              ┌─────────────────────┘                    │   │
│  │        │              ↓                                          │   │
│  │        │      [Retrieved Documents]                              │   │
│  │        │              │                                          │   │
│  │        ↓              ↓                                          │   │
│  │   ┌─────────────────────────┐                                    │   │
│  │   │     Prompt Template     │                                    │   │
│  │   │  Context: {docs}        │                                    │   │
│  │   │  Question: {query}      │                                    │   │
│  │   └───────────┬─────────────┘                                    │   │
│  │               ↓                                                  │   │
│  │           [LLM]  →  [Response]                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 컴포넌트 상세 설명

### 3.1 Document Loader (문서 로더)

다양한 형식의 문서를 시스템에 로드합니다.

| 문서 형식 | 설명 | 예시 라이브러리 |
|----------|------|----------------|
| PDF | PDF 문서 파싱 | PyPDF, PDFPlumber |
| HTML | 웹 페이지 크롤링 | BeautifulSoup |
| TXT | 텍스트 파일 | 기본 Python I/O |
| DOCX | Word 문서 | python-docx |
| CSV/JSON | 구조화된 데이터 | pandas |

### 3.2 Text Splitter (텍스트 분할기)

문서를 작은 청크(chunk)로 분할합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    원본 문서 (10,000 토큰)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    [ Text Splitter ]
                              ↓
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Chunk 1  │  │ Chunk 2  │  │ Chunk 3  │  │ Chunk 4  │
│ 500토큰  │  │ 500토큰  │  │ 500토큰  │  │ 500토큰  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

**주요 분할 전략:**
- **Fixed Size**: 고정 크기로 분할
- **Recursive**: 문단/문장 단위로 재귀적 분할
- **Semantic**: 의미 단위로 분할
- **Overlap**: 청크 간 중복 영역 설정 (문맥 유지)

**청킹 파라미터:**
- `chunk_size`: 청크 크기 (보통 500-1000 토큰)
- `chunk_overlap`: 청크 간 중복 (보통 50-200 토큰)

### 3.3 Embedding Model (임베딩 모델)

텍스트를 고차원 벡터로 변환합니다.

```
"RAG는 검색 증강 생성입니다"
            ↓
    [Embedding Model]
            ↓
[0.023, -0.156, 0.892, ..., 0.445]  (1536차원 벡터)
```

**주요 임베딩 모델:**
| 모델 | 제공자 | 차원 | 특징 |
|-----|-------|-----|------|
| text-embedding-ada-002 | OpenAI | 1536 | 범용, 높은 성능 |
| text-embedding-3-small | OpenAI | 1536 | 비용 효율적 |
| text-embedding-3-large | OpenAI | 3072 | 최고 성능 |
| all-MiniLM-L6-v2 | Sentence Transformers | 384 | 오픈소스, 가벼움 |
| multilingual-e5-large | Microsoft | 1024 | 다국어 지원 |

### 3.4 Vector Store (벡터 저장소)

임베딩 벡터를 저장하고 유사도 검색을 수행합니다.

**주요 벡터 DB:**
| DB | 유형 | 특징 |
|----|-----|------|
| Pinecone | 클라우드 | 완전 관리형, 확장성 |
| Weaviate | 오픈소스/클라우드 | 하이브리드 검색 |
| Chroma | 오픈소스 | 경량, 로컬 개발용 |
| Milvus | 오픈소스 | 대규모 처리 |
| FAISS | 라이브러리 | Facebook 개발, 고성능 |
| Qdrant | 오픈소스 | Rust 기반, 빠름 |

### 3.5 Retriever (검색기)

쿼리와 관련된 문서를 검색합니다.

**검색 방식:**
| 방식 | 설명 |
|-----|------|
| Dense Retrieval | 임베딩 벡터 간 코사인 유사도, 의미적 유사성 파악 |
| Sparse Retrieval | BM25, TF-IDF 기반 키워드 매칭 |
| Hybrid Retrieval | Dense + Sparse 결합, RRF 활용 |

**유사도 측정 방법:**
- **Cosine Similarity**: 벡터 방향의 유사도
- **Euclidean Distance**: 벡터 간 거리
- **Dot Product**: 내적 기반 유사도

### 3.6 LLM (Large Language Model)

검색된 컨텍스트를 바탕으로 응답을 생성합니다.

**주요 LLM 옵션:**
- OpenAI GPT-4, GPT-3.5
- Anthropic Claude
- Google Gemini
- Meta Llama (오픈소스)
- Mistral (오픈소스)

---

## 4. RAG 파이프라인 단계별 흐름

### Phase 1: Indexing (인덱싱)
1. **Load Documents**: PDF, HTML, TXT 등 문서 로드
2. **Split Text**: 문서를 청크로 분할
3. **Embed & Store**: 임베딩 후 벡터 DB에 저장

### Phase 2: Retrieval (검색)
1. **Query Input**: 사용자 질문 입력
2. **Query Embedding**: 질문을 벡터로 변환
3. **Similarity Search**: 유사한 문서 Top-K 검색

### Phase 3: Generation (생성)
1. **Prompt Construction**: 컨텍스트 + 질문으로 프롬프트 구성
2. **LLM Processing**: LLM이 답변 생성
3. **Response**: 사용자에게 답변 반환

---

## 5. 고급 RAG 기법

### 5.1 Query Transformation (쿼리 변환)
| 기법 | 설명 |
|-----|------|
| Query Rewriting | 쿼리를 더 명확하고 구체적으로 재작성 |
| HyDE | LLM이 가상의 답변 생성 → 답변을 임베딩하여 검색 |
| Multi-Query | 다양한 관점의 쿼리 생성 후 결과 병합 |

### 5.2 Re-Ranking (재순위화)
- 초기 검색 결과(Top 20)를 Cross-Encoder로 재순위화
- 최종 Top-K 선정으로 정확도 향상

### 5.3 Contextual Compression (컨텍스트 압축)
- 검색된 문서에서 쿼리 관련 핵심 내용만 추출
- 컨텍스트 크기 최적화

### 5.4 Self-RAG (자기 검증 RAG)
LLM이 스스로 검색 필요성과 답변 품질을 평가:
- `[Retrieve]`: 검색이 필요한가?
- `[IsRel]`: 검색 결과가 관련성 있는가?
- `[IsSup]`: 답변이 검색 결과에 의해 지원되는가?
- `[IsUse]`: 답변이 유용한가?

---

## 6. RAG vs Fine-Tuning 비교

| 항목 | RAG | Fine-Tuning |
|-----|-----|-------------|
| **데이터 업데이트** | 실시간 가능 | 재학습 필요 |
| **비용** | 인프라 비용 | GPU 학습 비용 |
| **출처 추적** | 가능 | 불가능 |
| **할루시네이션** | 감소 | 여전히 존재 |
| **도메인 적응** | 문서만 추가 | 데이터 수집 + 학습 |
| **적합한 경우** | 사실 기반 Q&A | 스타일/톤 변경 |

---

## 7. 주요 RAG 프레임워크

### 7.1 LangChain

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. 문서 로드
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 3. 임베딩 & 벡터 저장
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)

# 4. RAG 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 5. 질의
result = qa_chain({"query": "RAG의 장점은?"})
```

### 7.2 LlamaIndex

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

# 1. 문서 로드
documents = SimpleDirectoryReader("data").load_data()

# 2. 인덱스 생성 (자동으로 청킹, 임베딩, 저장)
index = VectorStoreIndex.from_documents(documents)

# 3. 쿼리 엔진 생성
query_engine = index.as_query_engine()

# 4. 질의
response = query_engine.query("RAG의 장점은?")
```

---

## 8. RAG 성능 최적화 팁

### 8.1 청킹 최적화
- 청크 크기는 300-500 토큰 권장
- 오버랩은 청크 크기의 10-20%
- 의미 단위로 분할 (문단, 섹션)

### 8.2 검색 최적화
- 하이브리드 검색 활용 (Dense + Sparse)
- Re-Ranker 적용
- Top-K 값 조정 (보통 3-5)

### 8.3 프롬프트 최적화
- 명확한 지시사항 포함
- 컨텍스트 활용 방법 명시
- Few-shot 예시 제공

### 8.4 평가 지표
- **Relevance**: 검색된 문서의 관련성
- **Faithfulness**: 답변이 컨텍스트에 충실한지
- **Answer Correctness**: 정답 정확도

---

## 9. 학습 로드맵

1. 기본 RAG 파이프라인 구현
2. 다양한 벡터 DB 실습
3. 청킹 전략 실험
4. 고급 기법 (Re-Ranking, Query Transformation) 적용
5. 평가 및 최적화

## 추천 리소스

- [LangChain 공식 문서](https://python.langchain.com/)
- [LlamaIndex 공식 문서](https://docs.llamaindex.ai/)
- [Pinecone Learning Center](https://www.pinecone.io/learn/)
- "RAG From Scratch" YouTube 시리즈 (LangChain)
